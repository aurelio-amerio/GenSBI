"""
Flow matching generative method strategy.

Implements :class:`~gensbi.core.generative_method.GenerativeMethod` using
optimal-transport conditional flow matching with an affine probability path.
"""

import jax
import jax.numpy as jnp

from gensbi.core.generative_method import GenerativeMethod
from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.loss import ContinuousFMLoss
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver


class _StandardNormal:
    """Default prior: standard normal distribution."""

    def sample(self, key, shape):
        return jax.random.normal(key, shape)


class FlowMatchingMethod(GenerativeMethod):
    """Flow matching strategy using affine probability paths.

    Uses the conditional optimal-transport scheduler and an ODE or SDE
    solver for sampling.

    Parameters
    ----------
    prior : object, optional
        Source distribution for noise sampling. Must implement
        ``sample(key, shape) -> Array``. Defaults to standard normal
        ``N(0, I)``.

    Examples
    --------
    >>> method = FlowMatchingMethod()
    >>> path = method.build_path(config={})
    >>> loss = method.build_loss(path)

    Using a custom prior:

    >>> from numpyro.distributions import Uniform
    >>> method = FlowMatchingMethod(prior=Uniform(low=-1.0, high=1.0))
    """

    def __init__(self, prior=None):
        self.prior = prior if prior is not None else _StandardNormal()

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
        _FMLoss
            A loss callable with uniform interface
            ``(model, batch, condition_mask=None, model_extras=None) -> loss``.
        """
        return _FMLoss(path)

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
            model_extras=model_extras,
            time_grid=time_grid,
        )

        def sampler_fn(key, x_init):
            if pass_key:
                key, key_sampler = jax.random.split(key)
                return sampler_(x_init, key_sampler)
            return sampler_(x_init)

        return sampler_fn


class _FMLoss:
    """Thin wrapper around ``ContinuousFMLoss`` for a uniform interface.

    Normalizes the ``ContinuousFMLoss(vf, batch, **kwargs)`` signature to
    ``(model, batch, condition_mask=None, model_extras=None)`` matching
    ``_EDMLoss`` and ``_SMLoss``.

    Parameters
    ----------
    path : AffineProbPath
        The probability path.
    """

    def __init__(self, path):
        self.loss = ContinuousFMLoss(path, reduction="mean")

    def __call__(self, model, batch, condition_mask=None, model_extras=None):
        """Evaluate the flow matching loss.

        Parameters
        ----------
        model : Callable
            The velocity field model.
        batch : tuple
            ``(x_0, x_1, t)`` — source noise, target data, and time.
        condition_mask : Array, optional
            Conditioning mask (forwarded as kwarg if not None).
        model_extras : dict, optional
            Additional model keyword arguments.

        Returns
        -------
        Array
            Scalar loss.
        """
        kwargs = {}
        if model_extras is not None:
            kwargs.update(model_extras)
        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask
        return self.loss(model, batch, **kwargs)
