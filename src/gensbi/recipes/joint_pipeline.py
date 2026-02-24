"""
Pipeline for training and using a Joint model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp

from gensbi.recipes.utils import init_ids_joint, build_edm_path, build_sm_path

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler
from gensbi.diffusion.solver import SMSolver, SMPFSolver

from einops import repeat

from gensbi.models import (
    JointCFMLoss,
    JointWrapper,
    JointEDMLoss,
)
from gensbi.models.losses import JointSMLoss

import numpyro.distributions as dist

from gensbi.utils.model_wrapping import _expand_dims

import os
import yaml

from gensbi.recipes.pipeline import AbstractPipeline, ModelEMA


def sample_structured_conditional_mask(
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
    Sample structured conditional masks for the Joint model.

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


def sample_condition_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    kind="structured",
):

    if kind == "structured":
        condition_mask = sample_structured_conditional_mask(
            key,
            num_samples,
            theta_dim,
            x_dim,
        )
    elif kind == "posterior":
        condition_mask = jnp.array(
            [False] * theta_dim + [True] * x_dim, dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    elif kind == "likelihood":
        condition_mask = jnp.array(
            [True] * theta_dim + [False] * x_dim, dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    elif kind == "joint":
        condition_mask = jnp.array(
            [False] * (theta_dim + x_dim), dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    else:
        raise ValueError(f"Unknown kind {kind} for condition mask.")

    return condition_mask


class JointFlowPipeline(AbstractPipeline):
    """
    Flow pipeline for training and using a Joint model for simulation-based inference.

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
        Number of channels for the observation space. Default is 1.
    params : JointParams, optional
        Parameters for the Joint model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.
    condition_mask_kind : str, optional
        Kind of condition mask to use. One of ["structured", "posterior"].

    Examples
    --------
    Minimal example on how to instantiate and use the JointFlowPipeline:

    .. literalinclude:: /examples/joint_flow_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/joint_flow_pipeline_marginals.png
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
        dim_cond: int,
        ch_obs=1,
        params=None,
        training_config=None,
        condition_mask_kind="structured",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.dim_joint = self.dim_obs + self.dim_cond

        # self.cond_ids = _expand_dims(self.cond_ids)
        # self.obs_ids = _expand_dims(self.obs_ids)
        # self.node_ids = _expand_dims(self.node_ids)

        self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(
            self.dim_obs, self.dim_cond
        )

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = JointCFMLoss(self.path)

        self.p0_joint = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_joint, self.ch_obs)),
                scale=jnp.ones((self.dim_joint, self.ch_obs)),
            ),
            reinterpreted_batch_ndims=2,
        )
        self.p0_obs = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_obs, self.ch_obs)),
                scale=jnp.ones((self.dim_obs, self.ch_obs)),
            ),
            reinterpreted_batch_ndims=2,
        )

        if self.dim_cond == 0:
            raise ValueError(
                "JointFlowPipeline initialized as unconditional since dim_cond=0. Please use `UnconditionalFlowPipeline` instead."
            )

        self.condition_mask_kind = condition_mask_kind

        if self.condition_mask_kind not in ["structured", "posterior"]:
            raise ValueError(
                f"condition_mask_kind must be one of ['structured', 'posterior'], got {self.condition_mask_kind}."
            )

    @classmethod
    def init_pipeline_from_config(cls):
        raise NotImplementedError(
            "init_pipeline_from_config is not implemented for JointFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "_make_model is not implemented for JointFlowPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for JointDiffusionPipeline."
        )

    def get_loss_fn(
        self,
    ):
        def loss_fn(
            model,
            x_1,
            key: jax.random.PRNGKey,
        ):
            batch_size = x_1.shape[0]
            rng_x0, rng_t, rng_condition = jax.random.split(key, 3)
            x_0 = self.p0_joint.sample(rng_x0, (batch_size,))
            t = jax.random.uniform(rng_t, x_1.shape[0])
            batch = (x_0, x_1, t)

            condition_mask = sample_condition_mask(
                rng_condition,
                batch_size,
                self.dim_obs,
                self.dim_cond,
                kind=self.condition_mask_kind,
            )

            loss = self.loss_fn(
                model,
                batch,
                node_ids=self.node_ids,
                condition_mask=condition_mask,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)
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
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            assert jnp.all(time_grid[:-1] <= time_grid[1:])
            return_intermediates = True

        cond = _expand_dims(x_o)

        solver_cls, solver_kwargs = solver
        solver_instance = solver_cls(velocity_model=model, **solver_kwargs)
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

        sampler = self.get_sampler(
            x_o,
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            solver=solver,
            **model_extras,
        )

        samples = sampler(key, nsamples)
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

    #     exact_log_p = logp_sampler(x_1)
    #     return exact_log_p


class JointDiffusionPipeline(AbstractPipeline):
    """
    Diffusion pipeline for training and using a Joint model for simulation-based inference.

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
        Number of channels for the observation space. Default is 1.
    params : optional
        Parameters for the Joint model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.
    condition_mask_kind : str, optional
        Kind of condition mask to use. One of ["structured", "posterior"].

    Examples
    --------
    Minimal example on how to instantiate and use the JointDiffusionPipeline:

    .. literalinclude:: /examples/joint_diffusion_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/joint_diffusion_pipeline_marginals.png
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
        dim_cond: int,
        ch_obs=1,
        params=None,
        training_config=None,
        condition_mask_kind="structured",
        sde="EDM",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.sde = sde

        # self.cond_ids = _expand_dims(self.cond_ids)
        # self.obs_ids = _expand_dims(self.obs_ids)
        # self.node_ids = _expand_dims(self.node_ids)

        self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(
            self.dim_obs, self.dim_cond
        )

        self.path = build_edm_path(sde, self.training_config)

        self.loss_fn = JointEDMLoss(self.path)

        if self.dim_cond == 0:
            raise ValueError(
                "JointFlowPipeline initialized as unconditional since dim_cond=0. Please use `UnconditionalFlowPipeline` instead."
            )

        self.condition_mask_kind = condition_mask_kind

        if self.condition_mask_kind not in ["structured", "posterior"]:
            raise ValueError(
                f"condition_mask_kind must be one of ['structured', 'posterior'], got {self.condition_mask_kind}."
            )

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "init_pipeline_from_config is not implemented for JointDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "_make_model is not implemented for JointDiffusionPipeline."
        )

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for JointDiffusionPipeline."
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
                    "beta_max": 19.9,
                }
            )
        return config

    def get_loss_fn(
        self,
    ):
        def loss_fn(
            model,
            x_1,
            key: jax.random.PRNGKey,
        ):
            batch_size = x_1.shape[0]

            rng_x0, rng_sigma, rng_condition = jax.random.split(key, 3)

            # sigma = self.path.sample_sigma(rng_sigma, x_1.shape[0])
            # sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")
            # sigma = self.path.sample_sigma(rng_sigma, (batch_size, self.dim_obs, self.ch_obs))
            # sigma = self.path.sample_sigma(rng_sigma, (batch_size,))
            sigma = self.path.sample_sigma(rng_sigma, (batch_size, 1, 1))

            batch = (x_1, sigma)

            condition_mask = sample_condition_mask(
                rng_condition,
                batch_size,
                self.dim_obs,
                self.dim_cond,
                kind=self.condition_mask_kind,
            )

            loss = self.loss_fn(
                rng_x0,
                model,
                batch,
                condition_mask=condition_mask,
                node_ids=self.node_ids,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)
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

        samples = sampler(key, nsamples)
        return samples


class JointSMPipeline(AbstractPipeline):
    """
    Score matching pipeline for training and using a Joint model for simulation-based inference.

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
    dim_cond : int
        Dimension of the observation space.
    ch_obs : int, optional
        Number of channels for the observation space. Default is 1.
    sde_type : str
        Type of SDE to use. One of ``"VP"`` (Variance Preserving) or ``"VE"`` (Variance Exploding).
    params : optional
        Parameters for the Joint model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.
    condition_mask_kind : str, optional
        Kind of condition mask to use. One of ["structured", "posterior"].
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs=1,
        sde_type: str = "VP",
        params=None,
        training_config=None,
        condition_mask_kind="structured",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.dim_joint = self.dim_obs + self.dim_cond

        self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(
            self.dim_obs, self.dim_cond
        )

        self.sde_type = sde_type

        self.path = build_sm_path(sde_type, self.training_config)

        self.loss_fn = JointSMLoss(self.path)

        if self.dim_cond == 0:
            raise ValueError(
                "JointSMPipeline initialized as unconditional since dim_cond=0. Please use `UnconditionalSMPipeline` instead."
            )

        self.condition_mask_kind = condition_mask_kind

        if self.condition_mask_kind not in ["structured", "posterior"]:
            raise ValueError(
                f"condition_mask_kind must be one of ['structured', 'posterior'], got {self.condition_mask_kind}."
            )

    @classmethod
    def init_pipeline_from_config(cls):
        raise NotImplementedError(
            "init_pipeline_from_config is not implemented for JointSMPipeline."
        )

    def _make_model(self):
        raise NotImplementedError("_make_model is not implemented for JointSMPipeline.")

    @classmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for JointSMPipeline."
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
        def loss_fn(
            model,
            x_1,
            key: jax.random.PRNGKey,
        ):
            batch_size = x_1.shape[0]

            rng_x0, rng_t, rng_condition = jax.random.split(key, 3)

            t = self.path.sample_t(rng_t, (batch_size, 1, 1))

            batch = (x_1, t)

            condition_mask = sample_condition_mask(
                rng_condition,
                batch_size,
                self.dim_obs,
                self.dim_cond,
                kind=self.condition_mask_kind,
            )

            loss = self.loss_fn(
                rng_x0,
                model,
                batch,
                condition_mask=condition_mask,
                node_ids=self.node_ids,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)
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

        samples = sampler(key, nsamples)
        return samples


# ---------------------------------------------------------------------------
# Unified JointPipeline (Phase 2)
# ---------------------------------------------------------------------------

from gensbi.core.generative_method import GenerativeMethod


class JointPipeline(AbstractPipeline):
    """Model-agnostic joint pipeline parameterized by a ``GenerativeMethod``.

    Unlike the method-specific classes above (``JointFlowPipeline``,
    ``JointDiffusionPipeline``, ``JointSMPipeline``), this class works with
    **any** generative method and **any** user-provided model that conforms
    to the ``JointWrapper`` interface.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : iterable
        Training dataset yielding concatenated ``x_1`` batches
        (obs and cond concatenated along the token dimension).
    val_dataset : iterable
        Validation dataset.
    dim_obs : int
        Dimension of the observation/parameter space.
    dim_cond : int
        Dimension of the conditioning space.
    method : GenerativeMethod
        Strategy object (e.g. ``FlowMatchingMethod()``,
        ``DiffusionEDMMethod()``, ``ScoreMatchingMethod()``).
    ch_obs : int, optional
        Number of channels per token. Default is 1.
    condition_mask_kind : str, optional
        Kind of condition mask. One of ``"structured"`` or ``"posterior"``.
        Default is ``"structured"``.
    params : optional
        Model parameters (stored but not used directly).
    training_config : dict, optional
        Training configuration.

    Examples
    --------
    >>> from gensbi.core import FlowMatchingMethod
    >>> pipeline = JointPipeline(
    ...     model=my_model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     dim_obs=2, dim_cond=7,
    ...     method=FlowMatchingMethod(),
    ... )
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        method: GenerativeMethod,
        ch_obs=1,
        condition_mask_kind="structured",
        params=None,
        training_config=None,
    ):
        self.method = method

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
            params=params,
            training_config=training_config,
        )

        self.dim_joint = self.dim_obs + self.dim_cond

        self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(
            self.dim_obs, self.dim_cond
        )

        self.path = method.build_path(self.training_config)
        self.loss_obj = method.build_loss(self.path)

        if self.dim_cond == 0:
            raise ValueError(
                "JointPipeline initialized with dim_cond=0. "
                "Use UnconditionalPipeline instead."
            )

        if condition_mask_kind not in ("structured", "posterior"):
            raise ValueError(
                f"condition_mask_kind must be one of ['structured', 'posterior'], "
                f"got {condition_mask_kind}."
            )
        self.condition_mask_kind = condition_mask_kind

    # -- Factory stubs ------------------------------------------------------

    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "JointPipeline is model-agnostic. "
            "Use model-specific pipelines for config init."
        )

    def _make_model(self):
        raise NotImplementedError(
            "JointPipeline is model-agnostic — the user provides the model."
        )

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "JointPipeline is model-agnostic — the user provides model params."
        )

    # -- Core pipeline methods ----------------------------------------------

    def get_loss_fn(self):
        def loss_fn(model, x_1, key):
            batch_size = x_1.shape[0]
            rng_batch, rng_condition = jax.random.split(key)

            prepared = self.method.prepare_batch(rng_batch, x_1, self.path)

            condition_mask = sample_condition_mask(
                rng_condition,
                batch_size,
                self.dim_obs,
                self.dim_cond,
                kind=self.condition_mask_kind,
            )

            model_extras = {
                "node_ids": self.node_ids,
                "condition_mask": condition_mask,
            }
            return self.loss_obj(
                model, prepared,
                condition_mask=condition_mask,
                model_extras=model_extras,
            )

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)

    def get_sampler(self, x_o, use_ema=True, **sampler_kwargs):
        """Get a sampler function.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (observed data).
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        **sampler_kwargs
            Forwarded to ``method.build_sampler_fn``.

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
