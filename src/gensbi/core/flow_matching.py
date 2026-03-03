"""
Flow matching generative method strategy.

Implements :class:`~gensbi.core.generative_method.GenerativeMethod` using
optimal-transport conditional flow matching with an affine probability path.
"""

import jax
import jax.numpy as jnp

import numpyro.distributions as dist

from gensbi.core.generative_method import GenerativeMethod
from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver

from gensbi.flow_matching.loss import FMLoss  


class StandardNormalPrior:
    """Default prior: standard normal distribution with ``log_prob`` support.

    Wraps ``jax.random.normal`` for sampling and uses
    ``numpyro.distributions.Independent(Normal(...))`` for ``log_prob``.
    The distribution is lazily constructed on the first ``log_prob`` call
    to match the shape of the input.
    """

    def sample(self, key, shape):
        return jax.random.normal(key, shape)

    def log_prob(self, x):
        """Compute log-probability under N(0, I) for the non-batch dims."""
        # x has shape (batch, features, channels) or (features, channels)
        event_shape = x.shape[-2:]  # (features, channels)
        p0 = dist.Independent(
            dist.Normal(
                loc=jnp.zeros(event_shape),
                scale=jnp.ones(event_shape),
            ),
            reinterpreted_batch_ndims=len(event_shape),
        )
        return p0.log_prob(x)


class FlowMatchingMethod(GenerativeMethod):
    """Flow matching strategy using affine probability paths.

    Uses the conditional optimal-transport scheduler and an ODE or SDE
    solver for sampling.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution, optional
        Source distribution. Must implement ``sample(key, shape)`` and
        ``log_prob(x)``. If you need log-probability evaluation (via
        ``build_log_prob_fn``), the prior **must** be a proper numpyro
        distribution. Defaults to ``StandardNormalPrior`` (≡ N(0, I)).

    Examples
    --------
    >>> method = FlowMatchingMethod()
    >>> path = method.build_path(config={})
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

    def __init__(self, prior=None):
        self.prior = prior if prior is not None else StandardNormalPrior()

    def build_path(self, config):
        """Build an affine probability path with the CondOT scheduler.

        Parameters
        ----------
        config : dict
            Training configuration (unused for flow matching).

        Returns
        -------
        AffineProbPath
            The probability path.
        """
        return AffineProbPath(scheduler=CondOTScheduler())

    def build_loss(self, path):
        """Build the continuous flow matching loss.

        Parameters
        ----------
        path : AffineProbPath
            The probability path.

        Returns
        -------
        FMLoss
            A loss callable with uniform interface
            ``(model, batch, condition_mask=None, model_extras=None) -> loss``.
        """
        return FMLoss(path)

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
            ``t`` is uniform in ``[0, 1)``.
        """
        rng_x0, rng_t = jax.random.split(key)
        x_0 = self.prior.sample(rng_x0, x_1.shape)
        t = jax.random.uniform(rng_t, (x_1.shape[0],))
        return (x_0, x_1, t)

    def get_default_solver(self):
        """Return the default ODE solver.

        Returns
        -------
        tuple
            ``(ODESolver, {})``
        """
        return (ODESolver, {})

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
            ``(SolverClass, kwargs)``. Defaults to ``(ODESolver, {})``.\

        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver
        return solver_cls(velocity_model=model_wrapped, **solver_kwargs)

    def sample_init(self, key, shape, path):
        """Sample from the prior distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        shape : tuple
            Shape of the initial sample.
        path
            The probability path (unused).

        Returns
        -------
        Array
            Sample from the prior.
        """
        return self.prior.sample(key, shape)

    def build_sampler_fn(self, model_wrapped, path, model_extras,
                         step_size=0.01, time_grid=None, solver=None,
                         **kwargs):
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
        pass_key = isinstance(solver_instance, BaseFmSDESolver)

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            return_intermediates = True

        sampler_ = solver_instance.get_sampler(
            method="Euler",
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
                          exact_divergence=True, **kwargs):
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

        if not isinstance(solver_instance, ODESolver):
            raise NotImplementedError(
                f"Log-probability computation requires ODESolver, "
                f"got {type(solver_instance).__name__}."
            )

        if time_grid is None:
            time_grid = [1.0, 0.0]

        # Get log_p0 from the prior
        log_p0 = self.prior.log_prob

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



