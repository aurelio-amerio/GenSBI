"""
Pipeline for training and using a Flux1Joint model for simulation-based inference.
"""

import jax.numpy as jnp
from flax import nnx

from gensbi.models import (
    Flux1Joint,
    Flux1JointParams,
)

import yaml

from gensbi.recipes.joint_pipeline import JointPipeline
from gensbi.recipes.utils import parse_training_config


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
        val_emb_dim=model_params.get(
            "val_emb_dim", model_params.get("val_emb_dim", 10)
        ),  # Support both for now
        cond_emb_dim=model_params.get("cond_emb_dim", 4),
        id_emb_dim=model_params.get("id_emb_dim", 10),
        qkv_bias=model_params.get("qkv_bias", True),
        id_merge_mode=model_params.get(
            "id_merge_mode", model_params.get("id_merge_mode", "concat")
        ),
        id_embedding_strategy=model_params.get(
            "id_embedding_strategy", model_params.get("id_embedding_strategy", "absolute")
        ),
        guidance_embed=model_params.get("guidance_embed", False),
        param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
    )

    return params_dict


def get_default_flux1joint_params(dim_joint: int, in_channels: int = 1):
    """
    Return default parameters for the Flux1Joint model.
    """
    return Flux1JointParams(
        in_channels=in_channels,
        vec_in_dim=None,
        mlp_ratio=3.0,
        num_heads=4,
        depth_single_blocks=8,
        val_emb_dim=10,
        cond_emb_dim=4,
        id_emb_dim=10,
        qkv_bias=True,
        rngs=nnx.Rngs(0),
        dim_joint=dim_joint,
        id_merge_mode="concat",
        guidance_embed=False,
        param_dtype=jnp.bfloat16,
    )


def _flux1joint_config_from_path(config_path: str, dim_joint: int):
    """
    Helper to parse common configuration for Flux1Joint pipelines.

    Returns
    -------
    params : Flux1JointParams
        The parsed model parameters.
    training_config : dict
        The parsed training configuration.
    method : str
        The methodology (flow or diffusion) specified in the config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # methodology
    strategy = config.get("strategy", {})
    method = strategy.get("method")
    model_type = strategy.get("model")

    if model_type != "flux1joint":
        raise ValueError(f"Model type {model_type} not supported.")

    params_dict = parse_flux1joint_params(config_path)

    params = Flux1JointParams(
        rngs=nnx.Rngs(0),
        dim_joint=dim_joint,
        **params_dict,
    )

    training_config = parse_training_config(config_path)

    return params, training_config, method


class Flux1JointFlowPipeline(JointPipeline):
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
        Flow pipeline for training and using a Flux1Joint model for simulation-based inference.

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
            Parameters for the Flux1Joint model. If None, default parameters are used.
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
            params = get_default_flux1joint_params(self.dim_joint, self.ch_obs)

        model = self._make_model(params)

        from gensbi.core import FlowMatchingMethod
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
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
        **kwargs,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        **kwargs
            Additional keyword arguments forwarded to the constructor.

        """
        params, training_config, method = _flux1joint_config_from_path(
            config_path, dim_obs + dim_cond
        )

        if method != "flow":
            raise ValueError(
                f"Method {method} not supported in Flux1JointFlowPipeline."
            )

        # add checkpoint dir to training config
        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1Joint model to be trained.
        """
        model = Flux1Joint(params)
        return model

    @classmethod
    def get_default_params(cls, dim_joint, in_channels):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1joint_params(dim_joint, in_channels)


class Flux1JointSMPipeline(JointPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs: int = 1,
        sde_type: str = "VP",
        params=None,
        training_config=None,
        condition_mask_kind="structured",
    ):
        """
        Score matching pipeline for training and using a Flux1Joint model for simulation-based inference.

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
        sde_type : str
            Type of SDE. One of ``"VP"`` or ``"VE"``.
        params : Flux1JointParams, optional
            Parameters for the Flux1Joint model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        condition_mask_kind : str, optional
            Kind of condition mask to use. One of ["structured", "posterior"].
        """
        self.dim_joint = dim_obs + dim_cond

        self.ch_obs = ch_obs

        if params is None:
            params = get_default_flux1joint_params(self.dim_joint, self.ch_obs)

        model = self._make_model(params)

        from gensbi.core import ScoreMatchingMethod
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=ScoreMatchingMethod(sde_type=sde_type),
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
        **kwargs,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        **kwargs
            Additional keyword arguments forwarded to the constructor
            (e.g. ``sde_type="VE"`` for score matching).
        """
        params, training_config, method = _flux1joint_config_from_path(
            config_path, dim_obs + dim_cond
        )

        if method != "score_matching":
            raise ValueError(
                f"Method {method} not supported in Flux1JointSMPipeline."
            )

        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1Joint model to be trained.
        """
        model = Flux1Joint(params)
        return model

    @classmethod
    def get_default_params(cls, dim_joint, in_channels):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1joint_params(dim_joint, in_channels)


class Flux1JointDiffusionPipeline(JointPipeline):
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
        Diffusion pipeline for training and using a Flux1Joint model for simulation-based inference.

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
            Parameters for the Flux1Joint model. If None, default parameters are used.
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
            params = get_default_flux1joint_params(self.dim_joint, self.ch_obs)

        model = self._make_model(params)

        from gensbi.core import DiffusionEDMMethod
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=DiffusionEDMMethod(),
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
        **kwargs,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        **kwargs
            Additional keyword arguments forwarded to the constructor.

        """
        params, training_config, method = _flux1joint_config_from_path(
            config_path, dim_obs + dim_cond
        )

        if method != "diffusion":
            raise ValueError(
                f"Method {method} not supported in Flux1JointDiffusionPipeline."
            )

        # add checkpoint dir to training config
        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1Joint model to be trained.
        """
        model = Flux1Joint(params)
        return model

    @classmethod
    def get_default_params(cls, dim_joint, in_channels):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1joint_params(dim_joint, in_channels)
