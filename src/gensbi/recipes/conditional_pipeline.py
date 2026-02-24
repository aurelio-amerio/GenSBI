"""
Pipeline for training and using a Conditional model for simulation-based inference.
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

from typing import Union, Tuple

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import (
    ODESolver,
    BaseFmSDESolver,
    ZeroEndsSolver,
    NonSingularSolver,
)

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler
from gensbi.diffusion.solver import SMSolver, SMPFSolver

from gensbi.models import ConditionalCFMLoss, ConditionalWrapper, ConditionalEDMLoss
from gensbi.models.losses import ConditionalSMLoss

from einops import repeat

from gensbi.models.flux1 import model
from gensbi.utils.model_wrapping import _expand_dims

import os

import yaml

from gensbi.recipes.pipeline import AbstractPipeline

from gensbi.recipes.utils import _resolve_embedding_ids, build_edm_path, build_sm_path


class ConditionalFlowPipeline(AbstractPipeline):
    """
    Flow pipeline for training and using a Conditional model for simulation-based inference.

    Parameters
    ----------
    model: nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int or tuple of int
        Dimension of the parameter space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    dim_cond : int or tuple of int
        Dimension of the observation space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    ch_obs : int, optional
        Number of channels per token in the observation data. Default is 1.
    ch_cond : int, optional
        Number of channels per token in the conditional data. Default is 1.
    params : ConditionalParams, optional
        Parameters for the Conditional model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the ConditionalFlowPipeline:

    .. literalinclude:: /examples/conditional_flow_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/conditional_flow_pipeline_marginals.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    .. note::
        Sampling in the latent space (latent diffusion/flow) is not currently supported.

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: Union[int, Tuple[int, int]],
        dim_cond: Union[int, Tuple[int, int]],
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        params=None,
        training_config=None,
    ):

        # if latent diffusion is enabled, make sure to adjust the dimensionality accordingly of the transformer model

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1
        )

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = ConditionalCFMLoss(self.path)

        # self.p0_obs = dist.Independent(
        #     dist.Normal(
        #         loc=jnp.zeros((self.dim_obs, self.ch_obs)),
        #         scale=jnp.ones((self.dim_obs, self.ch_obs)),
        #     ),
        #     reinterpreted_batch_ndims=2,
        # )

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for ConditionalFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for ConditionalFlowPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalDiffusionPipeline."
        )

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            # obs = batch[:, : self.dim_obs, ...]
            # cond = batch[:, self.dim_obs :, ...]
            obs, cond = batch
            rng_x0, rng_t = jax.random.split(key, 2)

            batch_size = obs.shape[0]

            x_1 = obs
            # x_0 = self.p0_obs.sample(rng_x0, (batch_size,))
            x_0 = jax.random.normal(rng_x0, (batch_size, self.dim_obs, self.ch_obs))
            t = jax.random.uniform(rng_t, x_1.shape[0])

            obs_batch = (x_0, x_1, t)

            loss = self.loss_fn(model, obs_batch, cond, self.obs_ids, self.cond_ids)
            return loss

        return loss_fn

    # need to change wrt
    # def _get_optimizer(self):
    #     """
    #     Construct the optimizer for training, including learning rate scheduling and gradient clipping.

    #     Returns
    #     -------
    #     optimizer : nnx.Optimizer
    #         The optimizer instance for the model.
    #     """

    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains('sbi_model'))
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

    #     opt = optax.chain(
    #         optax.adaptive_grad_clip(10.0),
    #         optax.adamw(self.training_config["max_lr"]),
    #         reduce_on_plateau(
    #             patience=self.training_config["patience"],
    #             cooldown=self.training_config["cooldown"],
    #             factor=self.training_config["factor"],
    #             rtol=self.training_config["rtol"],
    #             accumulation_size=self.training_config["accumulation_size"],
    #             min_scale=self.training_config["min_scale"],
    #         ),
    #     )
    #     if self.training_config["multistep"] > 1:
    #         opt = optax.MultiSteps(opt, self.training_config["multistep"])

    #     # optimizer = nnx.Optimizer(self.model, opt, wrt=sbi_model_params)
    #     optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
    #     return optimizer



    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        x_o,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the flow model.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (e.g., observed data).
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

            For SDE solvers (e.g., ``ZeroEndsSolver``, ``NonSingularSolver``), additional
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

        >>> sampler = pipeline.get_sampler(x_o)
        >>> samples = sampler(key, nsamples=1000)

        Using the ZeroEnds SDE solver:

        >>> import jax.numpy as jnp
        >>> from gensbi.flow_matching.solver import ZeroEndsSolver
        >>> solver = (ZeroEndsSolver, {
        ...     "mu0": jnp.zeros((dim_obs, ch_obs)),
        ...     "sigma0": jnp.ones((dim_obs, ch_obs)),
        ...     "alpha": 1.0,
        ... })
        >>> sampler = pipeline.get_sampler(x_o, solver=solver)
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

        cond = _expand_dims(x_o)

        solver_cls, solver_kwargs = solver
        solver_instance = solver_cls(velocity_model=vf_wrapped, **solver_kwargs)
        pass_key = isinstance(solver_instance, BaseFmSDESolver)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras,
        }

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
        x_o,
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
        x_o : array-like
            Conditioning variable (e.g., observed data).
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
            >>> samples = pipeline.sample(key, x_o, solver=solver)

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Array
            Sampled output of shape ``(nsamples, dim_obs, ch_obs)``.

        Examples
        --------
        Using the default ODE solver:

        >>> samples = pipeline.sample(key, x_o, nsamples=1000)

        Using the ZeroEnds SDE solver:

        >>> from gensbi.flow_matching.solver import ZeroEndsSolver
        >>> solver = (ZeroEndsSolver, {"mu0": 0.0, "sigma0": 0.01})
        >>> samples = pipeline.sample(key, x_o, solver=solver)
        """

        sampler_ = self.get_sampler(
            x_o,
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            solver=solver,
            **model_extras,
        )

        samples = sampler_(key, nsamples)

        return samples

    # def compute_unnorm_logprob(
    #     self, x_1, x_o, step_size=0.01, use_ema=True, time_grid=None, **model_extras
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
    #     cond = _expand_dims(x_o)

    #     model_extras = {
    #         "cond": cond,
    #         "obs_ids": self.obs_ids,
    #         "cond_ids": self.cond_ids,
    #         **model_extras,
    #     }

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


class ConditionalDiffusionPipeline(AbstractPipeline):
    """
    Diffusion pipeline for training and using a Conditional model for simulation-based inference.

    Parameters
    ----------
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int or tuple of int
        Dimension of the parameter space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    dim_cond : int or tuple of int
        Dimension of the observation space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    params : ConditionalParams, optional
        Parameters for the Conditional model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the ConditionalDiffusionPipeline:

    .. literalinclude:: /examples/conditional_diffusion_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/conditional_diffusion_pipeline_marginals.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    .. note::
        Sampling in the latent space (latent diffusion/flow) is not currently supported.

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: Union[int, Tuple[int, int]],
        dim_cond: Union[int, Tuple[int, int]],
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        sde="EDM",
        params=None,
        training_config=None,
    ):

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.sde = sde

        # # Flux1 uses different ids for obs and cond
        # obs_ids = jnp.zeros((1, dim_obs, 2), dtype=jnp.int32)
        # obs_ids = obs_ids.at[..., 0].set(jnp.arange(dim_obs))

        # cond_ids = jnp.zeros((1, dim_cond, 2), dtype=jnp.int32)
        # cond_ids = cond_ids.at[..., 0].set(jnp.arange(dim_cond))
        # cond_ids = cond_ids.at[..., 1].set(
        #     1
        # )  # set second channel to 1 for conditioning tokens

        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1
        )

        self.path = build_edm_path(sde, self.training_config)

        self.loss_fn = ConditionalEDMLoss(self.path)

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for ConditionalDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for ConditionalDiffusionPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalDiffusionPipeline."
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
                    "sigma_min": 0.02,
                    "sigma_max": 100.0,
                }
            )
        elif sde == "VP":
            config.update(
                {
                    "beta_min": 0.1,
                    "beta_max": 20.0,
                }
            )
        return config

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            # jax debug print(batch.shape)
            # (batch_size, dim_obs + dim_cond)

            # obs = jnp.take_along_axis(batch, self.obs_ids, axis=1)
            # cond = jnp.take_along_axis(batch, self.cond_ids, axis=1)
            # obs = batch[:, : self.dim_obs, ...]
            # cond = batch[:, self.dim_obs :, ...]

            obs, cond = batch

            rng_x0, rng_sigma = jax.random.split(key, 2)

            x_1 = obs
            # sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0],))
            sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], 1, 1))
            # sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")  # TODO fixme

            obs_batch = (x_1, sigma)
            loss = self.loss_fn(
                rng_x0, model, obs_batch, cond, self.obs_ids, self.cond_ids
            )
            return loss

        return loss_fn

    # def _get_optimizer(self):
    #     """
    #     Construct the optimizer for training, including learning rate scheduling and gradient clipping.

    #     Returns
    #     -------
    #     optimizer : nnx.Optimizer
    #         The optimizer instance for the model.
    #     """
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("sbi_model"))
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

    #     opt = optax.chain(
    #         optax.adaptive_grad_clip(10.0),
    #         optax.adamw(self.training_config["max_lr"]),
    #         reduce_on_plateau(
    #             patience=self.training_config["patience"],
    #             cooldown=self.training_config["cooldown"],
    #             factor=self.training_config["factor"],
    #             rtol=self.training_config["rtol"],
    #             accumulation_size=self.training_config["accumulation_size"],
    #             min_scale=self.training_config["min_scale"],
    #         ),
    #     )
    #     if self.training_config["multistep"] > 1:
    #         opt = optax.MultiSteps(opt, self.training_config["multistep"])

    #     # optimizer = nnx.Optimizer(self.model, opt, wrt=sbi_model_params)
    #     optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
    #     return optimizer



    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        x_o,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the diffusion model.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (e.g., observed data).
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

        >>> sampler = pipeline.get_sampler(x_o)
        >>> samples = sampler(key, nsamples=1000)

        Using a custom scheduler:

        >>> from gensbi.diffusion.solver import EDMSolver
        >>> from gensbi.diffusion.path.scheduler import VEEdmScheduler
        >>> solver = (EDMSolver, {"solver_scheduler": VEEdmScheduler()})
        >>> sampler = pipeline.get_sampler(x_o, solver=solver)
        >>> samples = sampler(key, nsamples=1000)
        """
        if solver is None:
            solver = (EDMSolver, {})

        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        cond = _expand_dims(x_o)

        solver_cls, solver_kwargs = solver
        # Extract EDM-specific options from solver_kwargs
        solver_scheduler = solver_kwargs.pop("solver_scheduler", None)
        solver_params = solver_kwargs.pop("solver_params", {})

        solver_instance = solver_cls(score_model=model, path=self.path, **solver_kwargs)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras,
        }

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
        x_o,
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
        x_o : array-like
            Conditioning variable (e.g., observed data).
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

        >>> samples = pipeline.sample(key, x_o, nsamples=1000)

        Using a custom scheduler:

        >>> from gensbi.diffusion.solver import EDMSolver
        >>> from gensbi.diffusion.path.scheduler import VEEdmScheduler
        >>> solver = (EDMSolver, {"solver_scheduler": VEEdmScheduler()})
        >>> samples = pipeline.sample(key, x_o, solver=solver)
        """

        sampler = self.get_sampler(
            x_o,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            solver=solver,
            **model_extras,
        )
        return sampler(key, nsamples)


class ConditionalSMPipeline(AbstractPipeline):
    """
    Score matching pipeline for training and using a Conditional model for simulation-based inference.

    Supports both Variance Preserving (VP) and Variance Exploding (VE) SDE formulations.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int or tuple of int
        Dimension of the parameter space (number of tokens).
    dim_cond : int or tuple of int
        Dimension of the observation space (number of tokens).
    ch_obs : int, optional
        Number of channels per token in the observation data. Default is 1.
    ch_cond : int, optional
        Number of channels per token in the conditional data. Default is 1.
    sde_type : str
        Type of SDE to use. One of ``"VP"`` (Variance Preserving) or ``"VE"`` (Variance Exploding).
    id_embedding_strategy : tuple of str, optional
        Embedding strategy for observation and conditioning IDs.
    params : optional
        Parameters for the model.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: Union[int, Tuple[int, int]],
        dim_cond: Union[int, Tuple[int, int]],
        ch_obs=1,
        ch_cond=1,
        sde_type: str = "VP",
        id_embedding_strategy=("absolute", "absolute"),
        params=None,
        training_config=None,
    ):

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1
        )
        self.sde_type = sde_type

        self.path = build_sm_path(sde_type, self.training_config)

        self.loss_fn = ConditionalSMLoss(self.path)

    @classmethod
    def init_pipeline_from_config(cls):
        raise NotImplementedError(
            "Initialization from config not implemented for ConditionalSMPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for ConditionalSMPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalSMPipeline."
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
            obs, cond = batch

            rng_x0, rng_t = jax.random.split(key, 2)

            x_1 = obs
            t = self.path.sample_t(rng_t, (x_1.shape[0], 1, 1))

            obs_batch = (x_1, t)
            loss = self.loss_fn(
                rng_x0, model, obs_batch, cond, self.obs_ids, self.cond_ids
            )
            return loss

        return loss_fn



    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        x_o,
        nsteps=1000,
        use_ema=True,
        return_intermediates=False,
        solver=None,
        **model_extras,
    ):
        """Get a sampler function for the score matching model.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (e.g., observed data).
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

        >>> sampler = pipeline.get_sampler(x_o)
        >>> samples = sampler(key, nsamples=1000)

        Using the Probability Flow ODE solver:

        >>> from gensbi.diffusion.solver import SMPFSolver
        >>> sampler = pipeline.get_sampler(x_o, solver=(SMPFSolver, {}))
        >>> samples = sampler(key, nsamples=1000)
        """
        if solver is None:
            solver = (SMSolver, {})

        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        cond = _expand_dims(x_o)

        solver_cls, solver_kwargs = solver
        solver_instance = solver_cls(score_model=model, path=self.path, **solver_kwargs)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras,
        }

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
        x_o,
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
        x_o : array-like
            Conditioning variable (e.g., observed data).
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
            >>> samples = pipeline.sample(key, x_o, solver=solver)

        **model_extras : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        Array
            Sampled output of shape ``(nsamples, dim_obs, ch_obs)``.

        Examples
        --------
        Using the default SDE solver:

        >>> samples = pipeline.sample(key, x_o, nsamples=1000)

        Using the probability flow ODE solver:

        >>> from gensbi.diffusion.solver import SMPFSolver
        >>> solver = (SMPFSolver, {})
        >>> samples = pipeline.sample(key, x_o, solver=solver)
        """
        sampler = self.get_sampler(
            x_o,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            solver=solver,
            **model_extras,
        )
        return sampler(key, nsamples)


# ---------------------------------------------------------------------------
# Unified ConditionalPipeline (Phase 2)
# ---------------------------------------------------------------------------

from gensbi.core.generative_method import GenerativeMethod


class ConditionalPipeline(AbstractPipeline):
    """Model-agnostic conditional pipeline parameterized by a ``GenerativeMethod``.

    Unlike the method-specific classes above (``ConditionalFlowPipeline``,
    ``ConditionalDiffusionPipeline``, ``ConditionalSMPipeline``), this class
    works with **any** generative method and **any** user-provided model that
    conforms to the ``ConditionalWrapper`` interface.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : iterable
        Training dataset yielding ``(obs, cond)`` batches.
    val_dataset : iterable
        Validation dataset yielding ``(obs, cond)`` batches.
    dim_obs : int or tuple of int
        Dimension of the observation/parameter space.
    dim_cond : int or tuple of int
        Dimension of the conditioning space.
    method : GenerativeMethod
        Strategy object (e.g. ``FlowMatchingMethod()``,
        ``DiffusionEDMMethod()``, ``ScoreMatchingMethod()``).
    ch_obs : int, optional
        Number of channels per observation token. Default is 1.
    ch_cond : int, optional
        Number of channels per conditioning token. Default is 1.
    id_embedding_strategy : tuple of str, optional
        Embedding strategy for observation and conditioning IDs.
        Default is ``("absolute", "absolute")``.
    params : optional
        Model parameters (stored but not used directly).
    training_config : dict, optional
        Training configuration. If ``None``, uses defaults augmented
        by ``method.get_extra_training_config()``.

    Examples
    --------
    >>> from gensbi.core import FlowMatchingMethod
    >>> pipeline = ConditionalPipeline(
    ...     model=my_model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     dim_obs=5, dim_cond=3,
    ...     method=FlowMatchingMethod(),
    ... )
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        method: GenerativeMethod,
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        params=None,
        training_config=None,
    ):
        self.method = method

        # Merge method-specific defaults before super().__init__ which
        # computes derived values from training_config.
        if training_config is None:
            training_config = self.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1
        )

        self.path = method.build_path(self.training_config)
        self.loss_obj = method.build_loss(self.path)

    # -- Factory stubs (model-agnostic: user provides model) ----------------

    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic. "
            "Use model-specific pipelines (e.g. Flux1FlowPipeline) for config init."
        )

    def _make_model(self):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic — the user provides the model."
        )

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic — the user provides model params."
        )

    # -- Core pipeline methods ----------------------------------------------

    def get_loss_fn(self):
        def loss_fn(model, batch, key):
            obs, cond = batch
            prepared = self.method.prepare_batch(key, obs, self.path)
            model_extras = {
                "cond": cond,
                "obs_ids": self.obs_ids,
                "cond_ids": self.cond_ids,
            }
            return self.loss_obj(model, prepared, model_extras=model_extras)

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)

    def get_sampler(self, x_o, use_ema=True, **sampler_kwargs):
        """Get a sampler function.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (observed data).
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        **sampler_kwargs
            Forwarded to ``method.build_sampler_fn`` (e.g. ``step_size``,
            ``nsteps``, ``solver``, ``time_grid``).

        Returns
        -------
        Callable
            ``sampler(key, nsamples) -> samples``
        """
        model_wrapped = self.ema_model_wrapped if use_ema else self.model_wrapped

        cond = _expand_dims(x_o)
        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
        }

        sampler_fn = self.method.build_sampler_fn(
            model_wrapped, self.path, model_extras, **sampler_kwargs,
        )

        def sampler(key, nsamples):
            key, key_init = jax.random.split(key)
            x_init = self.method.sample_init(
                key_init, (nsamples, self.dim_obs, self.ch_obs), self.path,
            )
            return sampler_fn(key, x_init)

        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : array-like
            Conditioning variable.
        nsamples : int, optional
            Number of samples. Default is 10 000.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """
        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return sampler(key, nsamples)
