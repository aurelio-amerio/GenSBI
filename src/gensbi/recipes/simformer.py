"""
Pipeline for training and using a Simformer model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import config, nnx

import yaml


from gensbi.models import (
    Simformer,
    SimformerParams,
)


from gensbi.recipes.joint_pipeline import JointFlowPipeline, JointDiffusionPipeline


def parse_simformer_params(config_path: str):
    """
    Parse a Simformer configuration file.

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
        dim_value=model_params.get("dim_value", 40),
        dim_id=model_params.get("dim_id", 40),
        dim_condition=model_params.get("dim_condition", 10),
        fourier_features=model_params.get("fourier_features", 128),
        num_heads=model_params.get("num_heads", 4),
        num_layers=model_params.get("num_layers", 8),
        widening_factor=model_params.get("widening_factor", 3),
        qkv_features=model_params.get("qkv_features", 90),
        num_hidden_layers=model_params.get("num_hidden_layers", 1),
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

    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)
    MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

    ema_decay = opt_params.get("ema_decay", 0.999)
    decay_transition = opt_params.get("decay_transition", 0.85)

    warmup_steps = opt_params.get("warmup_steps", 500)

    training_config = {}
    # overwrite the defaults with the config file values
    training_config["nsteps"] = nsteps
    training_config["ema_decay"] = ema_decay
    training_config["decay_transition"] = decay_transition

    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_SCALE
    training_config["val_every"] = val_every
    training_config["early_stopping"] = early_stopping
    training_config["experiment_id"] = experiment_id
    training_config["multistep"] = multistep
    training_config["warmup_steps"] = warmup_steps

    return training_config


class SimformerFlowPipeline(JointFlowPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
        edge_mask=None,
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
        params : SimformerParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        edge_mask : jnp.ndarray, optional
            Edge mask for the Simformer model. If None, no mask is applied.
        condition_mask_kind : str, optional
            Kind of condition mask to use. One of ["structured", "posterior"].

        Examples
        --------
        Minimal example on how to instantiate and use the SimformerFlowPipeline:

        .. literalinclude:: /examples/simformer_flow_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/simformer_flow_pipeline_marginals.png
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

        self.edge_mask = edge_mask

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
        ), f"Method {method} not supported in SimformerFlowPipeline."
        assert (
            model_type == "simformer"
        ), f"Model type {model_type} not supported in SimformerFlowPipeline."

        # Model parameters from config
        dim_joint = dim_obs + dim_cond

        params_dict = parse_simformer_params(config_path)

        params = SimformerParams(
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
        model = Simformer(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        params = SimformerParams(
            rngs=nnx.Rngs(0),
            in_channels=self.ch_obs,
            dim_value=40,
            dim_id=40,
            dim_condition=10,
            dim_joint=self.dim_joint,
            fourier_features=128,
            num_heads=4,
            num_layers=8,
            widening_factor=3,
            qkv_features=40,
            num_hidden_layers=1,
        )
        return params

    def sample(
        self, key, x_o, nsamples=10_000, step_size=0.01, use_ema=True, time_grid=None
    ):
        model_extras = {
            "edge_mask": self.edge_mask,
        }

        return super().sample(
            key,
            x_o,
            nsamples=nsamples,
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            **model_extras,
        )

    # def compute_unnorm_logprob(
    #     self, x_1, x_o, step_size=0.01, use_ema=True, time_grid=None
    # ):
    #     model_extras = {
    #         "edge_mask": self.edge_mask,
    #     }

    #     return super().compute_unnorm_logprob(
    #         x_1,
    #         x_o,
    #         step_size=step_size,
    #         use_ema=use_ema,
    #         time_grid=time_grid,
    #         **model_extras,
    #     )


class SimformerDiffusionPipeline(JointDiffusionPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
        edge_mask=None,
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
        params : SimformerParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.
        edge_mask : jnp.ndarray, optional
            Edge mask for the Simformer model. If None, no mask is applied.
        condition_mask_kind : str, optional
            Kind of condition mask to use. One of ["structured", "posterior"].

        Examples
        --------
        Minimal example on how to instantiate and use the SimformerDiffusionPipeline:

        .. literalinclude:: /examples/simformer_diffusion_pipeline.py
            :language: python
            :linenos:

        .. image:: /examples/simformer_diffusion_pipeline_marginals.png
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

        self.edge_mask = edge_mask

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
        ), f"Method {method} not supported in SimformerDiffusionPipeline."
        assert (
            model_type == "simformer"
        ), f"Model type {model_type} not supported in SimformerDiffusionPipeline."

        # Model parameters from config
        dim_joint = dim_obs + dim_cond

        params_dict = parse_simformer_params(config_path)

        params = SimformerParams(
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
        model = Simformer(params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        params = SimformerParams(
            in_channels=self.ch_obs,
            dim_value=40,
            dim_id=40,
            dim_condition=10,
            dim_joint=self.dim_joint,
            fourier_features=128,
            num_heads=4,
            num_layers=8,
            widening_factor=3,
            qkv_features=40,
            rngs=nnx.Rngs(0),
            num_hidden_layers=1,
        )
        return params

    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
    ):

        model_extras = {
            "edge_mask": self.edge_mask,
        }

        return super().sample(
            key,
            x_o,
            nsamples=nsamples,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            **model_extras,
        )
