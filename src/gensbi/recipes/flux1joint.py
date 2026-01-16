"""
Pipeline for training and using a Flux1Joint model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import (
    Flux1Joint,
    Flux1JointParams,
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
        id_embedding_strategy=model_params.get("id_embedding_strategy", "absolute"),
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

    RTOL = opt_params.get("rtol", 1e-4)
    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)
    MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

    warmup_steps = opt_params.get("warmup_steps", 500)

    ema_decay = opt_params.get("ema_decay", 0.999)

    decay_transition = opt_params.get("decay_transition", 0.85)

    training_config = {}
    # overwrite the defaults with the config file values
    training_config["nsteps"] = nsteps
    training_config["ema_decay"] = ema_decay
    training_config["decay_transition"] = decay_transition

    training_config["rtol"] = RTOL
    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_SCALE
    training_config["val_every"] = val_every
    training_config["early_stopping"] = early_stopping
    training_config["experiment_id"] = experiment_id
    training_config["multistep"] = multistep
    training_config["warmup_steps"] = warmup_steps

    return training_config


class Flux1JointFlowPipeline(JointFlowPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
        condition_mask_kind="structured",
    ):
        """
        Flow pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_obs : int
            Dimension of the parameter space.
        dim_cond : int
            Dimension of the observation space.
        ch_obs : int
            Number of channels in the observation data.
        params : Flux1JointParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        condition_mask_kind : str, optional
            Kind of condition mask to use. One of ["structured", "posterior"].

        Examples
        --------
        Minimal example on how to instantiate and use the Flux1JointFlowPipeline:

        .. literalinclude:: /examples/flux1joint_flow_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/flux1joint_flow_pipeline_marginals.png
            :width: 600

        .. note::
            If you plan on using multiprocessing prefetching, ensure that your script is wrapped
            in a ``if __name__ == "__main__":`` guard.
            See https://docs.python.org/3/library/multiprocessing.html

        """
        self.dim_joint = dim_obs + dim_cond

        self.ch_obs = ch_obs

        if params is None:
            params = self._get_default_params()

        model = self._make_model(params)

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
            condition_mask_kind=condition_mask_kind,
        )

        self.ema_model = nnx.clone(self.model)

    @classmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
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
        dim_joint = dim_obs + dim_cond

        params_dict = parse_flux1joint_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1JointParams(
            rngs=nnx.Rngs(0),
            dim_joint=dim_joint,
            **params_dict,
        )

        # Training parameters
        training_config = cls.get_default_training_config()
        training_config["checkpoint_dir"] = checkpoint_dir

        training_config_ = parse_training_config(config_path)

        for key, value in training_config_.items():
            training_config[key] = value  # update with config file values

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            params=params,
            training_config=training_config,
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
            in_channels=self.ch_obs,
            vec_in_dim=None,
            mlp_ratio=3.0,
            num_heads=4,
            depth_single_blocks=8,
            axes_dim=[10],
            condition_dim=[4],
            qkv_bias=True,
            rngs=nnx.Rngs(0),
            dim_joint=self.dim_joint,
            theta=self.dim_joint * 4,
            id_embedding_strategy="absolute",
            guidance_embed=False,
            param_dtype=jnp.bfloat16,
        )
        return params


class Flux1JointDiffusionPipeline(JointDiffusionPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
        condition_mask_kind="structured",
    ):
        """
        Diffusion pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_obs : int
            Dimension of the parameter space.
        dim_cond : int
            Dimension of the observation space.
        ch_obs : int
            Number of channels in the observation data.
        params : Flux1JointParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        condition_mask_kind : str, optional
            Kind of condition mask to use. One of ["structured", "posterior"]. Default is "structured".

        Examples
        --------
        Minimal example on how to instantiate and use the Flux1JointDiffusionPipeline:

        .. literalinclude:: /examples/flux1joint_diffusion_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/flux1joint_diffusion_pipeline_marginals.png
            :width: 600

        .. note::
            If you plan on using multiprocessing prefetching, ensure that your script is wrapped
            in a ``if __name__ == "__main__":`` guard.
            See https://docs.python.org/3/library/multiprocessing.html

        """
        self.dim_joint = dim_obs + dim_cond

        self.ch_obs = ch_obs

        if params is None:
            params = self._get_default_params()

        model = self._make_model(params)

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
            condition_mask_kind=condition_mask_kind,
        )

        self.ema_model = nnx.clone(self.model)

    @classmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
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
        dim_joint = dim_obs + dim_cond

        params_dict = parse_flux1joint_params(config_path)

        if params_dict["theta"] == -1:
            params_dict["theta"] = 4 * dim_joint

        params = Flux1JointParams(
            rngs=nnx.Rngs(0),
            dim_joint=dim_joint,
            **params_dict,
        )

        # Training parameters
        training_config = cls.get_default_training_config()
        training_config["checkpoint_dir"] = checkpoint_dir

        training_config_ = parse_training_config(config_path)

        for key, value in training_config_.items():
            training_config[key] = value  # update with config file values

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            params=params,
            training_config=training_config,
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
            in_channels=self.ch_obs,
            vec_in_dim=None,
            mlp_ratio=3.0,
            num_heads=4,
            depth_single_blocks=8,
            axes_dim=[10],
            condition_dim=[4],
            qkv_bias=True,
            rngs=nnx.Rngs(0),
            dim_joint=self.dim_joint,
            theta=self.dim_joint * 10,
            id_embedding_strategy="absolute",
            guidance_embed=False,
            param_dtype=jnp.bfloat16,
        )
        return params
