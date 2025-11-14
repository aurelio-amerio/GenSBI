"""
Pipeline for training and using a Flux1 model for simulation-based inference.

Example:
    .. code-block:: python
        import itertools
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import Flux1Pipeline

        # Define your training and validation datasets.
        train_data = jax.random.rand((1024, 4)) # your training dataset
        val_data = jax.random.rand((128, 4)) # your validation dataset

        batch_size = 32

        train_batch = train_data.reshape(-1, batch_size, train_data.shape[-1])
        val_batch = val_data.reshape(-1, batch_size, val_data.shape[-1])

        # Create datasets iterators (in this case with itertools, although a grain dataset is recommended)
        train_dataset = itertools.cycle(train_batch)
        val_dataset = itertools.cycle(val_batch)

        # Define the model
        dim_theta = 2  # Dimension of the parameter space
        dim_x = 2      # Dimension of the observation space
        pipeline = Flux1Pipeline(train_dataset, val_dataset, dim_theta, dim_x)

        # Train the model
        rngs = jax.random.PRNGKey(0)
        pipeline.train(rngs)

        # Sample from the posterior
        x_o = jnp.array([0.5, -0.2])  # Example
        samples = pipeline.sample(rngs, x_o, nsamples=10000, step_size=0.01)

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

from gensbi.models import Flux1, Flux1Params, ConditionalCFMLoss, ConditionalWrapper, ConditionalDiffLoss

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

import os

import yaml

from gensbi.recipes.conditional_pipeline import ConditionalFlowPipeline, ConditionalDiffusionPipeline


def parse_flux1_params(config_path: str):
    """
    Parse a Flux1 configuration file.

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
        context_in_dim=model_params.get("context_in_dim", 1),
        mlp_ratio=model_params.get("mlp_ratio", 4),
        num_heads=model_params.get("num_heads", 6),
        depth=model_params.get("depth", 8),
        depth_single_blocks=model_params.get("depth_single_blocks", 16),
        axes_dim=model_params.get("axes_dim", [6]),
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


class Flux1FlowPipeline(ConditionalFlowPipeline):
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
        Flow pipeline for training and using a Flux1 model for simulation-based inference.

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
        params : Flux1Params, optional
            Parameters for the Flux1 model. If None, default parameters are used.
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

        assert method == "flow", f"Method {method} not supported in Flux1FlowPipeline."
        assert (
            model_type == "flux"
        ), f"Model type {model_type} not supported in Flux1FlowPipeline."

        # Model parameters from config
        dim_joint = dim_theta + dim_x

        params_dict = parse_flux1_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1Params(
            rngs=nnx.Rngs(0),
            obs_dim=dim_theta,
            cond_dim=dim_x,
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
        Create and return the Flux1 model to be trained.
        """
        model = Flux1(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Flux1 model.
        """
        params = Flux1Params(
            in_channels=1,
            vec_in_dim=None,
            context_in_dim=1,
            mlp_ratio=4,
            num_heads=6,
            depth=8,
            depth_single_blocks=16,
            axes_dim=[6],
            qkv_bias=True,
            obs_dim=self.dim_theta,
            cond_dim=self.dim_x,
            theta=10 * (self.dim_theta + self.dim_x),
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
        )
        return params



class Flux1DiffusionPipeline(ConditionalDiffusionPipeline):
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
        Diffusion pipeline for training and using a Flux1 model for simulation-based inference.

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
        params : Flux1Params, optional
            Parameters for the Flux1 model. If None, default parameters are used.
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
        ), f"Method {method} not supported in Flux1DiffusionPipeline."
        assert (
            model_type == "flux"
        ), f"Model type {model_type} not supported in Flux1DiffusionPipeline."

        # Model parameters from config
        dim_joint = dim_theta + dim_x

        params_dict = parse_flux1_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1Params(
            rngs=nnx.Rngs(0),
            obs_dim=dim_theta,
            cond_dim=dim_x,
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
        Create and return the Flux1 model to be trained.
        """
        model = Flux1(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Flux1 model.
        """
        params = Flux1Params(
            in_channels=1,
            vec_in_dim=None,
            context_in_dim=1,
            mlp_ratio=4,
            num_heads=6,
            depth=8,
            depth_single_blocks=16,
            axes_dim=[6],
            qkv_bias=True,
            obs_dim=self.dim_theta,
            cond_dim=self.dim_x,
            theta=10 * (self.dim_theta + self.dim_x),
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
        )
        return params