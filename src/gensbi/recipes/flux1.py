"""
Pipeline for training and using a Flux1 model for simulation-based inference.
"""

import jax.numpy as jnp
from flax import nnx


from gensbi.models import (
    Flux1,
    Flux1Params,
)

import yaml

from gensbi.recipes.conditional_pipeline import ConditionalPipeline
from gensbi.recipes.utils import parse_training_config


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
        # Fields with set defaults in Flux1Params
        axes_dim=model_params.get("axes_dim", None),
        val_emb_dim=model_params.get("val_emb_dim", None),
        id_emb_dim=model_params.get("id_emb_dim", None),
        id_merge_mode=model_params.get("id_merge_mode", "sum"),
        qkv_bias=model_params.get("qkv_bias", True),
        theta=model_params.get("theta", None),
        id_embedding_strategy=tuple(
            model_params.get("id_embedding_strategy", ("absolute", "absolute"))
        ),
        param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
    )

    return params_dict


def get_default_flux1_params(
    dim_obs: int, dim_cond: int, ch_obs: int = 1, ch_cond: int = 1
) -> Flux1Params:
    """
    Return default parameters for the Flux1 model.
    """
    return Flux1Params(
        in_channels=ch_obs,
        vec_in_dim=None,
        context_in_dim=ch_cond,
        mlp_ratio=4,
        num_heads=6,
        depth=8,
        depth_single_blocks=16,
        qkv_bias=True,
        rngs=nnx.Rngs(default=42),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        axes_dim=[6, 0],
        theta=10 * (dim_obs + dim_cond),
        id_embedding_strategy=("absolute", "absolute"),
        param_dtype=jnp.float32,
    )


def _flux1_config_from_path(config_path: str, dim_obs: int, dim_cond: int):
    """
    Helper to parse common configuration for Flux1 pipelines.

    Returns
    -------
    params : Flux1Params
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

    if model_type != "flux":
        raise ValueError(f"Model type {model_type} not supported.")

    params_dict = parse_flux1_params(config_path)

    # Handle theta default logic if it was set to -1 (meaning "auto")
    if params_dict["theta"] in [-1, None]:
        dim_joint = dim_obs + dim_cond
        params_dict["theta"] = 10 * dim_joint  # Default value used in original code

    params = Flux1Params(
        rngs=nnx.Rngs(0),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        **params_dict,
    )

    training_config = parse_training_config(config_path)

    return params, training_config, method


class Flux1FlowPipeline(ConditionalPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs=1,
        ch_cond=1,
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
        dim_obs : int
            Dimension of the parameter space.
        dim_cond : int
            Dimension of the observation space.
        ch_obs : int, optional
            Number of channels in the observation data. Default is 1.
        ch_cond : int, optional
            Number of channels in the conditional data. Default is 1.
        params : Flux1Params, optional
            Parameters for the Flux1 model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        Examples
        --------
        Minimal example on how to instantiate and use the Flux1FlowPipeline:

        .. literalinclude:: /examples/flux1_flow_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/flux1_flow_pipeline_marginals.png
            :width: 600

        .. note::
            If you plan on using multiprocessing prefetching, ensure that your script is wrapped
            in a ``if __name__ == "__main__":`` guard.
            See https://docs.python.org/3/library/multiprocessing.html

        """

        if params is not None:
            ch_obs = params.in_channels

        if params is not None:
            ch_cond = params.context_in_dim

        if params is None:
            params = get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)

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
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
            id_embedding_strategy=params.id_embedding_strategy,
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
        params, training_config, method = _flux1_config_from_path(
            config_path, dim_obs, dim_cond
        )

        if method != "flow":
            raise ValueError(f"Method {method} not supported in Flux1FlowPipeline.")

        # add checkpoint dir to training config
        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            ch_cond=params.context_in_dim,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1 model to be trained.
        """
        model = Flux1(params)
        return model

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)


class Flux1DiffusionPipeline(ConditionalPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs=1,
        ch_cond=1,
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
        dim_obs : int
            Dimension of the parameter space.
        dim_cond : int
            Dimension of the observation space.
        ch_obs : int, optional
            Number of channels in the observation data. Default is 1.
        ch_cond : int, optional
            Number of channels in the conditional data. Default is 1.
        params : Flux1Params, optional
            Parameters for the Flux1 model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        Examples
        --------
        Minimal example on how to instantiate and use the Flux1DiffusionPipeline:

        .. literalinclude:: /examples/flux1_diffusion_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/flux1_diffusion_pipeline_marginals.png
            :width: 600

        .. note::
            If you plan on using multiprocessing prefetching, ensure that your script is wrapped
            in a ``if __name__ == "__main__":`` guard.
            See https://docs.python.org/3/library/multiprocessing.html

        """

        if params is not None:
            ch_obs = params.in_channels

        if params is not None:
            ch_cond = params.context_in_dim

        if params is None:
            params = get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)

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
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
            id_embedding_strategy=params.id_embedding_strategy,
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
        params, training_config, method = _flux1_config_from_path(
            config_path, dim_obs, dim_cond
        )

        if method != "diffusion":
            raise ValueError(
                f"Method {method} not supported in Flux1DiffusionPipeline."
            )

        # add checkpoint dir to training config
        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            ch_cond=params.context_in_dim,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1 model to be trained.
        """
        model = Flux1(params)
        return model

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)


class Flux1SMPipeline(ConditionalPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs=1,
        ch_cond=1,
        sde_type: str = "VP",
        params=None,
        training_config=None,
    ):
        """
        Score matching pipeline for training and using a Flux1 model for simulation-based inference.

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
        ch_obs : int, optional
            Number of channels in the observation data. Default is 1.
        ch_cond : int, optional
            Number of channels in the conditional data. Default is 1.
        sde_type : str
            Type of SDE. One of ``"VP"`` or ``"VE"``.
        params : Flux1Params, optional
            Parameters for the Flux1 model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        """

        if params is not None:
            ch_obs = params.in_channels

        if params is not None:
            ch_cond = params.context_in_dim

        if params is None:
            params = get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)

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
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
            id_embedding_strategy=params.id_embedding_strategy,
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
        params, training_config, method = _flux1_config_from_path(
            config_path, dim_obs, dim_cond
        )

        if method != "score_matching":
            raise ValueError(
                f"Method {method} not supported in Flux1SMPipeline."
            )

        training_config["checkpoint_dir"] = checkpoint_dir

        pipeline = cls(
            train_dataset,
            val_dataset,
            dim_obs,
            dim_cond,
            ch_obs=params.in_channels,
            ch_cond=params.context_in_dim,
            params=params,
            training_config=training_config,
            **kwargs,
        )

        return pipeline

    def _make_model(self, params):
        """
        Create and return the Flux1 model to be trained.
        """
        model = Flux1(params)
        return model

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        """
        Return a dictionary of default model parameters.
        """
        return get_default_flux1_params(dim_obs, dim_cond, ch_obs, ch_cond)
