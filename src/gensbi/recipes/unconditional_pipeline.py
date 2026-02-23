"""
Pipeline for training and using a Unconditional model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from numpyro import distributions as dist


from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler
from gensbi.diffusion.solver import SMSolver, SMPFSolver

from gensbi.models import (
    UnconditionalCFMLoss,
    UnconditionalWrapper,
    UnconditionalEDMLoss,
)
from gensbi.models.losses import UnconditionalSMLoss

from gensbi.recipes.utils import init_ids_1d, build_edm_path, build_sm_path

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

from gensbi.recipes.pipeline import AbstractPipeline


class UnconditionalFlowPipeline(AbstractPipeline):
    """
    Flow pipeline for training and using an Unconditional model for simulation-based inference.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int
        Dimension of the parameter space.
    ch_obs : int
        Number of channels in the observation space.
    params : optional
        Parameters for the model. Serves no use if a custom model is provided.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the UnconditionalFlowPipeline:

    .. literalinclude:: /examples/unconditional_flow_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/unconditional_flow_samples.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = init_ids_1d(self.dim_obs)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = UnconditionalCFMLoss(self.path)

        self.p0_obs = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_obs, self.ch_obs)),
                scale=jnp.ones((self.dim_obs, self.ch_obs)),
            ),
            reinterpreted_batch_ndims=2,
        )

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for UnconditionalFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for UnconditionalFlowPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, ch_obs):
        raise NotImplementedError(
            "Default parameters not implemented for UnconditionalFlowPipeline."
        )

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            obs = batch

            batch_size = batch.shape[0]

            rng_x0, rng_t = jax.random.split(key, 2)

            x_1 = obs
            # x_0 = self.p0_obs.sample(rng_x0, (batch_size,))
            x_0 = jax.random.normal(rng_x0, (batch_size, self.dim_obs, self.ch_obs))
            t = jax.random.uniform(rng_t, x_1.shape[0])

            batch = (x_0, x_1, t)
            condition_mask = jnp.zeros((*x_1.shape[:-1], 1), dtype=jnp.bool_)

            loss = self.loss_fn(
                model, batch, node_ids=self.obs_ids, condition_mask=condition_mask
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the flow model.

        Parameters
        ----------
        step_size : float, optional
            Step size for the solver. Default is 0.01.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        time_grid : Array, optional
            Time grid for intermediate steps. If None, uses ``[0, 1]``.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its constructor keyword arguments.
            Defaults to ``(ODESolver, {})``.

            For SDE solvers (e.g., ``ZeroEnds``, ``NonSingular``), additional
            constructor arguments must be provided via the kwargs dict.
            A random key is automatically passed to SDE samplers.

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Callable
            A function ``sampler(key, nsamples)`` that generates samples.

        Examples
        --------
        Using the default ODE solver:

        >>> sampler = pipeline.get_sampler()
        >>> samples = sampler(key, nsamples=1000)

        Using the ZeroEnds SDE solver:

        >>> import jax.numpy as jnp
        >>> from gensbi.flow_matching.solver import ZeroEnds
        >>> solver = (ZeroEnds, {
        ...     "mu0": jnp.zeros((dim_obs, ch_obs)),
        ...     "sigma0": jnp.ones((dim_obs, ch_obs)),
        ...     "alpha": 1.0,
        ... })
        >>> sampler = pipeline.get_sampler(solver=solver)
        >>> samples = sampler(key, nsamples=1000)
        """
        if solver is None:
            solver = (ODESolver, {})

        if use_ema:
            vf_wrapped = self.ema_model_wrapped
        else:
            vf_wrapped = self.model_wrapped

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            assert jnp.all(time_grid[:-1] <= time_grid[1:])
            return_intermediates = True

        solver_cls, solver_kwargs = solver
        solver_instance = solver_cls(velocity_model=vf_wrapped, **solver_kwargs)
        pass_key = isinstance(solver_instance, BaseFmSDESolver)

        model_extras = {"obs_ids": self.obs_ids, **model_extras}

        sampler_ = solver_instance.get_sampler(
            method="Euler",
            step_size=step_size,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
            time_grid=time_grid,
        )

        def sampler(key, nsamples):
            key, key_init = jax.random.split(key)
            x_init = jax.random.normal(key_init, (nsamples, self.dim_obs, self.ch_obs))

            if pass_key:
                key, key_sampler = jax.random.split(key)
                samples = sampler_(x_init, key_sampler)
            else:
                samples = sampler_(x_init)

            return samples

        return sampler

    def sample(
        self,
        key,
        nsamples=10_000,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        solver=None,
        **model_extras,
    ):
        """Draw samples from the flow model.

        Convenience method that internally calls :meth:`get_sampler` and
        immediately evaluates the returned sampler function.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key used for sampling.
        nsamples : int, optional
            Number of samples to draw. Default is 10 000.
        step_size : float, optional
            Step size for the solver. Default is 0.01.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        time_grid : Array, optional
            Time grid for intermediate steps. If None, uses
            ``jnp.linspace(0, 1, int(1 / step_size) + 1)``.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its keyword arguments. Defaults to ``(ODESolver, {})``.

            To use an SDE solver instead:

            >>> from gensbi.flow_matching.solver import ZeroEndsSolver
            >>> solver = (ZeroEndsSolver, {"mu0": 0.0, "sigma0": 0.01})
            >>> samples = pipeline.sample(key, solver=solver)

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Array
            Sampled output of shape ``(nsamples, dim_obs, ch_obs)``.

        Examples
        --------
        Using the default ODE solver:

        >>> samples = pipeline.sample(key, nsamples=1000)

        Using the ZeroEnds SDE solver:

        >>> from gensbi.flow_matching.solver import ZeroEndsSolver
        >>> solver = (ZeroEndsSolver, {"mu0": 0.0, "sigma0": 0.01})
        >>> samples = pipeline.sample(key, solver=solver)
        """
        sampler = self.get_sampler(
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            solver=solver,
            **model_extras,
        )
        samples = sampler(key, nsamples)
        return samples

    def sample_batched(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalFlowPipeline."
        )

    # def compute_unnorm_logprob(
    #     self, x_1, step_size=0.01, use_ema=True, time_grid=None, **model_extras
    # ):
    #     if use_ema:
    #         model = self.ema_model_wrapped
    #     else:
    #         model = self.model_wrapped

    #     if time_grid is None:
    #         time_grid = jnp.array([1.0, 0.0])
    #         return_intermediates = False
    #     else:
    #         # assert time grid is decreasing
    #         assert jnp.all(time_grid[:-1] >= time_grid[1:])
    #         return_intermediates = True

    #     solver = ODESolver(velocity_model=model)

    #     # x_1 = _expand_dims(x_1)
    #     assert (
    #         x_1.ndim == 2
    #     ), "x_1 must be of shape (num_samples, dim_obs), currently sampling for multiple channels is not supported."

    #     # todo need to check the model extras, is that node_ids instead?
    #     model_extras = {"obs_ids": self.obs_ids, **model_extras}

    #     logp_sampler = solver.get_unnormalized_logprob(
    #         time_grid=time_grid,
    #         method="Euler",
    #         step_size=step_size,
    #         log_p0=self.p0_obs.log_prob,
    #         model_extras=model_extras,
    #         return_intermediates=return_intermediates,
    #     )

    #     if len(x_1) > 4:
    #         # we trigger precompilation first
    #         _ = logp_sampler(x_1[:4])

    #     exact_log_p = logp_sampler(x_1)
    #     return exact_log_p


class UnconditionalDiffusionPipeline(AbstractPipeline):
    """
    Diffusion pipeline for training and using an Unconditional model for simulation-based inference.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int
        Dimension of the parameter space.
    ch_obs : int
        Number of channels in the observation space.
    params : optional
        Parameters for the model. Serves no use if a custom model is provided.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the UnconditionalDiffusionPipeline:

    .. literalinclude:: /examples/unconditional_diffusion_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/unconditional_diffusion_samples.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        ch_obs: int = 1,
        params=None,
        sde="EDM",
        training_config=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = init_ids_1d(self.dim_obs)

        self.sde = sde
        self.path = build_edm_path(sde, self.training_config)

        self.loss_fn = UnconditionalEDMLoss(self.path)

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for UnconditionalDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for UnconditionalDiffusionPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, ch_obs):
        raise NotImplementedError(
            "Default parameters not implemented for UnconditionalDiffusionPipeline."
        )

    @classmethod
    def get_default_training_config(cls, sde="EDM"):
        config = super().get_default_training_config()
        if sde == "EDM":
            config.update(
                {
                    "sigma_min": 0.002,  # from edm paper
                    "sigma_max": 80.0,
                }
            )
        elif sde == "VE":
            config.update(
                {
                    "sigma_min": 0.02,  # from edm paper
                    "sigma_max": 100.0,
                }
            )
        elif sde == "VP":
            config.update(
                {
                    "beta_min": 0.1,
                    "beta_max": 19.9,
                }
            )
        return config

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            rng_x0, rng_sigma = jax.random.split(key, 2)

            x_1 = batch
            # sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], ))
            sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], 1, 1))
            # sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")  # TODO fixme

            batch = (x_1, sigma)
            loss = self.loss_fn(rng_x0, model, batch, node_ids=self.obs_ids)
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the diffusion model.

        Parameters
        ----------
        nsteps : int, optional
            Number of sampling steps. Default is 18.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its constructor keyword arguments.
            Defaults to ``(EDMSolver, {})``.

            The solver class must accept ``score_model`` and ``path`` as
            its first two positional arguments.

            EDM-specific options can be provided in the kwargs dict:

            - ``solver_scheduler``: override the path's scheduler for
              sampling (also used for ``sample_prior``).
            - ``solver_params``: dict of EDM solver parameters
              (``S_churn``, ``S_min``, ``S_max``, ``S_noise``).

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Callable
            A function ``sampler(key, nsamples)`` that generates samples.

        Examples
        --------
        Using the default EDM solver:

        >>> sampler = pipeline.get_sampler()
        >>> samples = sampler(key, nsamples=1000)

        Using a custom scheduler:

        >>> from gensbi.diffusion.solver import EDMSolver
        >>> from gensbi.diffusion.path.scheduler import VEEdmScheduler
        >>> solver = (EDMSolver, {"solver_scheduler": VEEdmScheduler()})
        >>> sampler = pipeline.get_sampler(solver=solver)
        >>> samples = sampler(key, nsamples=1000)
        """
        if solver is None:
            solver = (EDMSolver, {})

        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        solver_cls, solver_kwargs = solver
        solver_scheduler = solver_kwargs.pop("solver_scheduler", None)
        solver_params = solver_kwargs.pop("solver_params", {})

        solver_instance = solver_cls(score_model=model, path=self.path, **solver_kwargs)

        model_extras = {"obs_ids": self.obs_ids, **model_extras}

        sampler_ = solver_instance.get_sampler(
            nsteps=nsteps,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
            solver_scheduler=solver_scheduler,
            solver_params=solver_params,
        )

        prior_source = solver_scheduler if solver_scheduler is not None else self.path

        def sampler(key, nsamples):
            key1, key2 = jax.random.split(key, 2)
            x_init = prior_source.sample_prior(
                key1, (nsamples, self.dim_obs, self.ch_obs)
            )
            samples = sampler_(key2, x_init)
            return samples

        return sampler

    def sample(
        self,
        key,
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Draw samples from the diffusion model.

        Convenience method that internally calls :meth:`get_sampler` and
        immediately evaluates the returned sampler function.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key used for sampling.
        nsamples : int, optional
            Number of samples to draw. Default is 10 000.
        nsteps : int, optional
            Number of sampling steps. Default is 18.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its keyword arguments. Defaults to ``(EDMSolver, {})``.

            EDM-specific options can be provided in the kwargs dict:

            - ``solver_scheduler``: override the path's scheduler for
              sampling (also used for ``sample_prior``).
            - ``solver_params``: dict of EDM solver parameters
              (``S_churn``, ``S_min``, ``S_max``, ``S_noise``).

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Array
            Sampled output of shape ``(nsamples, dim_obs, ch_obs)``.

        Examples
        --------
        Using the default EDM solver:

        >>> samples = pipeline.sample(key, nsamples=1000)

        Using a custom scheduler:

        >>> from gensbi.diffusion.solver import EDMSolver
        >>> from gensbi.diffusion.path.scheduler import VEEdmScheduler
        >>> solver = (EDMSolver, {"solver_scheduler": VEEdmScheduler()})
        >>> samples = pipeline.sample(key, solver=solver)
        """
        sampler = self.get_sampler(
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            solver=solver,
            **model_extras,
        )
        samples = sampler(key, nsamples)

        return samples

    def sample_batched(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalDiffusionPipeline."
        )


class UnconditionalSMPipeline(AbstractPipeline):
    """
    Score matching pipeline for training and using an Unconditional model for simulation-based inference.

    Supports both Variance Preserving (VP) and Variance Exploding (VE) SDE formulations.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int
        Dimension of the parameter space.
    ch_obs : int
        Number of channels in the observation space.
    sde_type : str
        Type of SDE to use. One of ``"VP"`` (Variance Preserving) or ``"VE"`` (Variance Exploding).
    params : optional
        Parameters for the model. Serves no use if a custom model is provided.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        ch_obs: int = 1,
        sde_type: str = "VP",
        params=None,
        training_config=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = init_ids_1d(self.dim_obs)
        self.sde_type = sde_type

        self.path = build_sm_path(sde_type, self.training_config)

        self.loss_fn = UnconditionalSMLoss(self.path)

    @classmethod
    def init_pipeline_from_config(cls):
        raise NotImplementedError(
            "Initialization from config not implemented for UnconditionalSMPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for UnconditionalSMPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, ch_obs):
        raise NotImplementedError(
            "Default parameters not implemented for UnconditionalSMPipeline."
        )

    @classmethod
    def get_default_training_config(cls, sde_type="VP"):
        config = super().get_default_training_config()
        if sde_type == "VP":
            config.update(
                {
                    "beta_min": 0.001,
                    "beta_max": 3.0,
                }
            )
        elif sde_type == "VE":
            config.update(
                {
                    "sigma_min": 0.001,
                    "sigma_max": 15.0,
                }
            )
        return config

    def get_loss_fn(self):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            rng_x0, rng_t = jax.random.split(key, 2)

            x_1 = batch
            t = self.path.sample_t(rng_t, (x_1.shape[0], 1, 1))

            batch = (x_1, t)
            loss = self.loss_fn(rng_x0, model, batch, node_ids=self.obs_ids)
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        nsteps=1000,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the score matching model.

        Parameters
        ----------
        nsteps : int, optional
            Number of integration steps. Default is 1000.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its constructor keyword arguments.
            Defaults to ``(SMSolver, {})``.

            The solver class must accept ``score_model`` and ``path`` as
            its first two positional arguments.

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Callable
            A function ``sampler(key, nsamples)`` that generates samples.

        Examples
        --------
        Using the default reverse SDE solver:

        >>> sampler = pipeline.get_sampler()
        >>> samples = sampler(key, nsamples=1000)

        Using the Probability Flow ODE solver:

        >>> from gensbi.diffusion.solver import SMPFSolver
        >>> sampler = pipeline.get_sampler(solver=(SMPFSolver, {}))
        >>> samples = sampler(key, nsamples=1000)
        """
        if solver is None:
            solver = (SMSolver, {})

        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        solver_cls, solver_kwargs = solver
        solver_instance = solver_cls(score_model=model, path=self.path, **solver_kwargs)

        model_extras = {"obs_ids": self.obs_ids, **model_extras}

        sampler_ = solver_instance.get_sampler(
            nsteps=nsteps,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
        )

        def sampler(key, nsamples):
            key1, key2 = jax.random.split(key, 2)
            x_init = self.path.sample_prior(key1, (nsamples, self.dim_obs, self.ch_obs))
            samples = sampler_(key2, x_init)
            return samples

        return sampler

    def sample(
        self,
        key,
        nsamples=10_000,
        nsteps=1000,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Draw samples from the score matching model.

        Convenience method that internally calls :meth:`get_sampler` and
        immediately evaluates the returned sampler function.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key used for sampling.
        nsamples : int, optional
            Number of samples to draw. Default is 10 000.
        nsteps : int, optional
            Number of sampling steps. Default is 1000.
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            A tuple ``(SolverClass, solver_kwargs)`` specifying the solver
            class and its keyword arguments. Defaults to ``(SMSolver, {})``.

            To use the probability flow ODE solver instead:

            >>> from gensbi.diffusion.solver import SMPFSolver
            >>> solver = (SMPFSolver, {})
            >>> samples = pipeline.sample(key, solver=solver)

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Array
            Sampled output of shape ``(nsamples, dim_obs, ch_obs)``.

        Examples
        --------
        Using the default SDE solver:

        >>> samples = pipeline.sample(key, nsamples=1000)

        Using the probability flow ODE solver:

        >>> from gensbi.diffusion.solver import SMPFSolver
        >>> solver = (SMPFSolver, {})
        >>> samples = pipeline.sample(key, solver=solver)
        """
        sampler = self.get_sampler(
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            solver=solver,
            **model_extras,
        )
        samples = sampler(key, nsamples)
        return samples

    def sample_batched(self, *args, **kwargs):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalSMPipeline."
        )
