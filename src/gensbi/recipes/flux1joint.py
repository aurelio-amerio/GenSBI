"""
Pipeline for training and using a Flux1 model for simulation-based inference.

Example:
    .. code-block:: python
        import grain
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import SimformerPipeline

        # Define your training and validation datasets.
        train_data = jax.random.rand((1024, 4)) # your training dataset
        val_data = jax.random.rand((128, 4)) # your validation dataset

        batch_size = 32

        train_dataset_grain = (
            grain.MapDataset.source(np.array(train_data)[...,None])
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
        )

        val_dataset_grain = (
            grain.MapDataset.source(np.array(val_data)[...,None])
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size) 
            # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
        )

        # Define the model
        dim_theta = 2  # Dimension of the parameter space
        dim_x = 2      # Dimension of the observation space
        pipeline = SimformerPipeline(train_dataset, val_dataset, dim_theta, dim_x)

        # Train the model
        rngs = jax.random.PRNGKey(0)
        pipeline.train(rngs)

        # Sample from the posterior
        x_o = jnp.array([0.5, -0.2])  # Example
        samples = pipeline.sample(rngs, x_o, nsamples=10000, step_size=0.01)

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped in a `if __name__ == "__main__":` guard. See https://docs.python.org/3/library/multiprocessing.html

"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau
from numpyro import distributions as dist
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEScheduler
from gensbi.diffusion.solver import SDESolver

from einops import repeat

from gensbi.models import (
    Flux1Joint,
    Flux1JointParams,
    JointCFMLoss,
    JointWrapper,
    JointDiffLoss,
)

import numpyro.distributions as dist

from gensbi.utils.model_wrapping import _expand_dims

import os
import yaml

from gensbi.recipes.joint_pipeline import JointFlowPipeline, JointDiffusionPipeline


def parse_flux1joint_params(config_path: str):
    """
    Parse a Flux1Joint configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.

    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_params = config.get("model", {})

    params_dict = dict(
        in_channels=model_params.get("in_channels", 1),
        vec_in_dim=model_params.get("vec_in_dim", None),
        mlp_ratio=model_params.get("mlp_ratio", 3.0),
        num_heads=model_params.get("num_heads", 4),
        depth_single_blocks=model_params.get("depth_single_blocks", 8),
        axes_dim=model_params.get("axes_dim", [10]),
        condition_dim=model_params.get("condition_dim", [4]),
        qkv_bias=model_params.get("qkv_bias", True),
        theta=model_params.get("theta", -1),
        param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
    )

    return params_dict


def parse_training_config(config_path: str):
    """
    Parse a training configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.

    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Training parameters
    train_params = config.get("training", {})
    multistep = train_params.get("multistep", 1)
    experiment_id = train_params.get("experiment_id", 1)
    early_stopping = train_params.get("early_stopping", True)
    nsteps = train_params.get("nsteps", 30000) * multistep
    val_every = train_params.get("val_every", 100) * multistep

    # Optimizer parameters
    opt_params = config.get("optimizer", {})
    PATIENCE = opt_params.get("patience", 10)
    COOLDOWN = opt_params.get("cooldown", 2)
    FACTOR = opt_params.get("factor", 0.5)
    ACCUMULATION_SIZE = opt_params.get("accumulation_size", 100) * multistep
    RTOL = opt_params.get("rtol", 1e-4)
    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)
    MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

    ema_decay = opt_params.get("ema_decay", 0.999)

    training_config = {}
    # overwrite the defaults with the config file values
    training_config["num_steps"] = nsteps
    training_config["ema_decay"] = ema_decay
    training_config["patience"] = PATIENCE
    training_config["cooldown"] = COOLDOWN
    training_config["factor"] = FACTOR
    training_config["accumulation_size"] = ACCUMULATION_SIZE
    training_config["rtol"] = RTOL
    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_SCALE
    training_config["val_every"] = val_every
    training_config["early_stopping"] = early_stopping
    training_config["experiment_id"] = experiment_id
    training_config["multistep"] = multistep

    return training_config


def sample_strutured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    """
    Sample structured conditional masks for the Simformer model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for sampling.
    num_samples : int
        Number of samples to generate.
    theta_dim : int
        Dimension of the parameter space.
    x_dim : int
        Dimension of the observation space.
    p_joint : float
        Probability of selecting the joint mask.
    p_posterior : float
        Probability of selecting the posterior mask.
    p_likelihood : float
        Probability of selecting the likelihood mask.
    p_rnd1 : float
        Probability of selecting the first random mask.
    p_rnd2 : float
        Probability of selecting the second random mask.
    rnd1_prob : float
        Probability of a True value in the first random mask.
    rnd2_prob : float
        Probability of a True value in the second random mask.

    Returns
    -------
    condition_mask : jnp.ndarray
        Array of shape (num_samples, theta_dim + x_dim) with boolean masks.

    """
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    joint_mask = jnp.array([False] * (theta_dim + x_dim), dtype=jnp.bool_)
    posterior_mask = jnp.array([False] * theta_dim + [True] * x_dim, dtype=jnp.bool_)
    likelihood_mask = jnp.array([True] * theta_dim + [False] * x_dim, dtype=jnp.bool_)
    random1_mask = jax.random.bernoulli(
        key2, rnd1_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    random2_mask = jax.random.bernoulli(
        key3, rnd2_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    mask_options = jnp.stack(
        [joint_mask, posterior_mask, likelihood_mask, random1_mask, random2_mask],
        axis=0,
    )  # (5, theta_dim + x_dim)
    idx = jax.random.choice(
        key1,
        5,
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
    )
    condition_mask = mask_options[idx]
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask[..., None]


class Flux1JointFlowPipeline(JointFlowPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Flow pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_theta : int
            Dimension of the parameter space.
        dim_x : int
            Dimension of the observation space.
        params : Flux1JointParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        
        super().__init__(
            None, train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )
        if params is None:
            self.params = self._get_default_params()
        
        self.model = self._make_model(self.params)
        self.ema_model = nnx.clone(self.model)

    @classmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        config_path: str,
        checkpoint_dir: str,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.

        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # methodology
        strategy = config.get("strategy", {})
        method = strategy.get("method")
        model_type = strategy.get("model")

        assert (
            method == "flow"
        ), f"Method {method} not supported in Flux1JointDiffusionPipeline."
        assert (
            model_type == "flux1joint"
        ), f"Model type {model_type} not supported in Flux1JointDiffusionPipeline."

        # Model parameters from config
        dim_joint = dim_theta + dim_x

        params_dict = parse_flux1joint_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1JointParams(
            rngs=nnx.Rngs(0),
            joint_dim=dim_joint,
            **params_dict,
        )

        # Training parameters
        training_config = cls._get_default_training_config()
        training_config["checkpoint_dir"] = checkpoint_dir

        training_config_ = parse_training_config(config_path)

        for key, value in training_config_.items():
            training_config[key] = value  # update with config file values

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_theta,
            dim_x,
            params,
            training_config,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Simformer model to be trained.
        """
        model = Flux1Joint(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        # TODO
        params = Flux1JointParams(
            in_channels=1,
            vec_in_dim=None,
            mlp_ratio=3.0,
            num_heads=4,
            depth_single_blocks=8,
            axes_dim=[10],
            condition_dim=[4],
            qkv_bias=True,
            rngs=nnx.Rngs(0),
            joint_dim=self.dim_joint,
            theta=self.dim_joint * 4,
            guidance_embed=False,
            param_dtype=jnp.bfloat16,
        )
        return params


class Flux1JointDiffusionPipeline(JointDiffusionPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Diffusion pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_theta : int
            Dimension of the parameter space.
        dim_x : int
            Dimension of the observation space.
        params : Flux1JointParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            None, train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )
        if params is None:
            self.params = self._get_default_params()
        
        self.model = self._make_model(self.params)
        self.ema_model = nnx.clone(self.model)

    @classmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        config_path: str,
        checkpoint_dir: str,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.

        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # methodology
        strategy = config.get("strategy", {})
        method = strategy.get("method")
        model_type = strategy.get("model")

        assert (
            method == "diffusion"
        ), f"Method {method} not supported in Flux1JointDiffusionPipeline."
        assert (
            model_type == "flux1joint"
        ), f"Model type {model_type} not supported in Flux1JointDiffusionPipeline."

        # Model parameters from config
        dim_joint = dim_theta + dim_x

        params_dict = parse_flux1joint_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1JointParams(
            rngs=nnx.Rngs(0),
            joint_dim=dim_joint,
            **params_dict,
        )

        # Training parameters
        training_config = cls._get_default_training_config()
        training_config["checkpoint_dir"] = checkpoint_dir

        training_config_ = parse_training_config(config_path)

        for key, value in training_config_.items():
            training_config[key] = value  # update with config file values

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_theta,
            dim_x,
            params,
            training_config,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Simformer model to be trained.
        """
        model = Flux1Joint(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        params = Flux1JointParams(
            in_channels=1,
            vec_in_dim=None,
            mlp_ratio=3.0,
            num_heads=4,
            depth_single_blocks=8,
            axes_dim=[10],
            condition_dim=[4],
            qkv_bias=True,
            rngs=nnx.Rngs(0),
            joint_dim=self.dim_joint,
            theta=self.dim_joint * 4,
            guidance_embed=False,
            param_dtype=jnp.bfloat16,
        )
        return params
