"""
Flow matching generative method strategy.

Implements :class:`~gensbi.core.generative_method.GenerativeMethod` using
optimal-transport conditional flow matching with an affine probability path.
"""

import jax
import jax.numpy as jnp

import numpyro.distributions as dist

from gensbi.core.generative_method import GenerativeMethod
from gensbi.core.prior import make_gaussian_prior, is_gaussian_prior
from gensbi.core.sde_solver import SDESolver
from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver.fm_ode_solver import FMODESolver

from gensbi.flow_matching.loss import FMLoss
from gensbi.core.time_sampling import sample_time


class FlowMatchingMethod(GenerativeMethod):
    """Flow matching strategy using affine probability paths.

    Uses the conditional optimal-transport scheduler and an ODE or SDE
    solver for sampling.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution, optional
        Source distribution. Must implement ``sample(key, shape)`` and
        ``log_prob(x)``. Validated against ``event_shape`` in
        :meth:`build_path`. If ``None``, a standard normal prior is
        constructed automatically.
    time_dist : str, optional
        Training-time timestep distribution used in :meth:`prepare_batch`.
        ``"uniform"`` (default) is bit-identical to the previous behaviour;
        ``"logitnormal"`` draws ``sigmoid(logitnorm_mean + logitnorm_std * N(0, 1))``.
    logitnorm_mean, logitnorm_std : float, optional
        Mean/std of the underlying normal for ``time_dist="logitnormal"``.

    Examples
    --------
    >>> method = FlowMatchingMethod()
    >>> path = method.build_path(config={}, event_shape=(5, 1))
    >>> loss = method.build_loss(path)

    Using a custom numpyro prior (x has shape ``(batch, dim_obs, ch_obs)``):

    >>> import numpyro.distributions as dist
    >>> dim_obs, ch_obs = 3, 1
    >>> prior = dist.Independent(
    ...     dist.Normal(loc=jnp.zeros((dim_obs, ch_obs)), scale=jnp.ones((dim_obs, ch_obs))),
    ...     reinterpreted_batch_ndims=2,
    ... )
    >>> method = FlowMatchingMethod(prior=prior)
    """

    def __init__(self, prior=None, *, time_dist="uniform",
                 logitnorm_mean=0.0, logitnorm_std=1.0):
        if time_dist not in ("uniform", "logitnormal"):
            raise ValueError(
                f"time_dist must be 'uniform' or 'logitnormal', got {time_dist!r}"
            )
        self._user_prior = prior
        self.prior = None
        self.time_dist = time_dist
        self.logitnorm_mean = float(logitnorm_mean)
        self.logitnorm_std = float(logitnorm_std)

    def build_path(self, config, event_shape):
        """Build an affine probability path with the CondOT scheduler.

        Also constructs or validates ``self.prior``.

        Parameters
        ----------
        config : dict
            Training configuration (unused for flow matching).
        event_shape : tuple of (int, int)
            ``(dim, ch)`` — feature and channel dimensions.

        Returns
        -------
        AffineProbPath
            The probability path.

        Raises
        ------
        ValueError
            If a user-supplied prior has a mismatched ``event_shape``.
        """
        if self._user_prior is not None:
            if self._user_prior.event_shape != event_shape:
                raise ValueError(
                    f"Prior event_shape {self._user_prior.event_shape} does not "
                    f"match expected {event_shape}."
                )
            self.prior = self._user_prior
        else:
            self.prior = make_gaussian_prior(tuple(event_shape))
        return AffineProbPath(scheduler=CondOTScheduler())

    def build_loss(self, path, weights=None):
        """Build the continuous flow matching loss.

        Parameters
        ----------
        path : AffineProbPath
            The probability path.
        weights : Array, optional
            Per-dimension loss weights.

        Returns
        -------
        FMLoss
            A loss callable with uniform interface
            ``(model, batch, condition_mask=None, model_extras=None) -> loss``.
        """
        return FMLoss(path, weights=weights)

    def prepare_batch(self, key, x_1, path):
        """Sample from the prior and time for a flow matching training batch.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_1 : Array
            Clean data of shape ``(batch_size, dim, ch)``.
        path : AffineProbPath
            The probability path (unused, kept for interface consistency).

        Returns
        -------
        tuple
            ``(x_0, x_1, t)`` where ``x_0`` is drawn from the prior and
            ``t`` follows ``time_dist`` (uniform in ``[0, 1)`` by default).
        """
        rng_x0, rng_t = jax.random.split(key)
        x_0 = self.prior.sample(rng_x0, (x_1.shape[0],))
        t = sample_time(
            rng_t, x_1.shape[0],
            dist=self.time_dist,
            logitnorm_mean=self.logitnorm_mean,
            logitnorm_std=self.logitnorm_std,
        )
        return (x_0, x_1, t)

    def get_default_solver(self):
        """Return the default ODE solver.

        Returns
        -------
        tuple
            ``(FMODESolver, {})``
        """
        return (FMODESolver, {})

    def build_solver(self, model_wrapped, path, solver=None):
        """Instantiate a flow matching solver.

        Supports both ODE solvers (``ODESolver``) and SDE solvers
        (``ZeroEndsSolver``, ``NonSingularSolver``).

        Parameters
        ----------
        model_wrapped
            The wrapped velocity field model.
        path
            The probability path (unused by ODE solver, but may be
            needed by SDE solvers).
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(ODESolver, {})``.\n
        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver

        if issubclass(solver_cls, SDESolver):
            if not is_gaussian_prior(self.prior):
                raise ValueError("FM SDE solvers require a Gaussian prior.")
            # Prior provides default mu0/sigma0; user kwargs override
            # (needed for joint pipeline where solver operates in obs-space)
            sde_kwargs = {
                "mu0": self.prior.base_dist.loc,
                "sigma0": self.prior.base_dist.scale,
            }
            sde_kwargs.update(solver_kwargs)
            return solver_cls(velocity_model=model_wrapped, **sde_kwargs)

        return solver_cls(velocity_model=model_wrapped, **solver_kwargs)

    def sample_init(self, key, nsamples):
        """Sample from the prior distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        nsamples : int
            Number of samples to draw.

        Returns
        -------
        Array
            Sample from the prior.
        """
        return self.prior.sample(key, (nsamples,))

    def build_sampler_fn(self, model_wrapped, path, model_extras,
                         step_size=0.01, method="Euler", time_grid=None,
                         solver=None, **kwargs):
        """Build a sampler closure for flow matching.

        Supports ODE solvers (deterministic) and SDE solvers (stochastic;
        ``ZeroEndsSolver``, ``NonSingularSolver``). When an SDE solver is
        used, the sampler function accepts and splits an extra random key.

        Parameters
        ----------
        model_wrapped
            The wrapped velocity field model.
        path
            The probability path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, ``cond_ids``, etc.).
        step_size : float, optional
            Step size for fixed-step solvers. Default is 0.01.
        method : str or diffrax solver, optional
            Integration method for the ODE/SDE solver. Default is ``"Euler"``.
            Other commonly used solvers are ``"Dopri5"`` (adaptive),
            ``diffrax.Heun()``, and ``diffrax.Midpoint()``.
        time_grid : Array, optional
            Time grid for integration. If ``None``, uses ``[0, 1]``.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(ODESolver, {})``.

        Returns
        -------
        sampler_fn : Callable
            A function ``(key, x_init) -> samples``.
        """
        solver_instance = self.build_solver(model_wrapped, path, solver=solver)
        pass_key = isinstance(solver_instance, SDESolver)

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            return_intermediates = True

        sampler_ = solver_instance.get_sampler(
            method=method,
            step_size=step_size,
            return_intermediates=return_intermediates,
            time_grid=time_grid,
        )

        def sampler_fn(key, x_init, model_extras=None):
            if model_extras is None:
                model_extras = {}
            if pass_key:
                key, key_sampler = jax.random.split(key)
                return sampler_(x_init, key_sampler, model_extras=model_extras)
            return sampler_(x_init, model_extras=model_extras)

        return sampler_fn

    def build_log_prob_fn(self, model_wrapped, path, model_extras,
                          step_size=0.01, method="Dopri5", atol=1e-5,
                          rtol=1e-5, time_grid=None, solver=None,
                          exact_divergence=True, log_prior=None, **kwargs):
        """Build a log-probability closure for flow matching.

        Uses the continuous change-of-variables formula via ``ODESolver``.
        Only works with ODE solvers (not SDE solvers).

        Parameters
        ----------
        model_wrapped
            The wrapped velocity field model.
        path
            The probability path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, etc.).
        step_size : float, optional
            Step size for fixed-step solvers. Default is 0.01.
        method : str or diffrax solver, optional
            Integration method. Default is ``"Dopri5"``.
        atol : float, optional
            Absolute tolerance for adaptive solvers.
        rtol : float, optional
            Relative tolerance for adaptive solvers.
        time_grid : list, optional
            Time grid. Defaults to ``[1.0, 0.0]``.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Must be an ODE solver.
        exact_divergence : bool, optional
            If ``True`` (default), compute exact divergence via full
            Jacobian. If ``False``, use the Hutchinson estimator (requires
            a PRNG ``key`` at call time).
        log_prior : callable, optional
            Override for the prior's ``log_prob``. If ``None``, uses
            ``self.prior.log_prob``. Used by the joint pipeline to pass
            a user-supplied obs-space prior.

        Returns
        -------
        log_prob_fn : Callable
            ``(x_1, model_extras, *, key=None) -> log_prob``.

        Raises
        ------
        NotImplementedError
            If a non-ODE solver is specified.
        """
        solver_instance = self.build_solver(model_wrapped, path, solver=solver)

        if not isinstance(solver_instance, FMODESolver):
            raise NotImplementedError(
                f"Log-probability computation requires FMODESolver, "
                f"got {type(solver_instance).__name__}."
            )

        if time_grid is None:
            time_grid = jnp.array([1.0, 0.0])

        # Use the provided log_prior if given, otherwise fall back to the prior
        log_p0 = log_prior if log_prior is not None else self.prior.log_prob

        log_prob_closure = solver_instance.get_log_prob(
            log_p0=log_p0,
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            exact_divergence=exact_divergence,
        )

        def log_prob_fn(x_1, model_extras=None, *, key=None):
            if model_extras is None:
                model_extras = {}
            return log_prob_closure(x_1, model_extras=model_extras, key=key)

        return log_prob_fn



