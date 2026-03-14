"""
Score matching generative method strategy.

Implements :class:`~gensbi.core.generative_method.GenerativeMethod` using
standard score matching diffusion with reverse SDE or probability flow ODE
solvers and support for VP and VE SDE formulations.
"""

import jax
import jax.numpy as jnp

from gensbi.core.generative_method import GenerativeMethod
from gensbi.recipes.utils import build_sm_path
from gensbi.diffusion.solver.sm_ode_solver_new import NewSMODESolver
from gensbi.diffusion.solver.sm_sde_solver_new import NewSMSDESolver

from gensbi.diffusion.loss import SMLoss
from gensbi.core.prior import make_gaussian_prior
from gensbi.utils.model_wrapping import ModelWrapper, ScoreToODEDrift


class ScoreMatchingMethod(GenerativeMethod):
    """Score matching strategy.

    Supports two SDE formulations via the ``sde_type`` parameter:

    - ``"VP"`` — variance-preserving (default)
    - ``"VE"`` — variance-exploding

    Sampling can use either the reverse SDE (``SMSDESolver``, default) or
    the probability flow ODE (``SMODESolver`` via ``ScoreToODEDrift``).

    Parameters
    ----------
    sde_type : str, optional
        SDE type. One of ``"VP"`` or ``"VE"``. Default is ``"VP"``.

    Examples
    --------
    >>> method = ScoreMatchingMethod(sde_type="VP")
    >>> path = method.build_path(config={"beta_min": 0.001, "beta_max": 3.0})
    >>> loss = method.build_loss(path)
    """

    def __init__(self, sde_type="VP"):
        if sde_type not in ("VP", "VE"):
            raise ValueError(f"sde_type must be one of 'VP', 'VE', got '{sde_type}'.")
        self.sde_type = sde_type
        self.prior = None

    def build_path(self, config, event_shape):
        """Build a score matching path.

        Also constructs ``self.prior`` as a numpyro distribution.

        Parameters
        ----------
        config : dict
            Training configuration. Reads scheduler hyperparameters
            (``beta_min``, ``beta_max`` for VP; ``sigma_min``, ``sigma_max``
            for VE) with sensible defaults.
        event_shape : tuple of (int, int)
            ``(dim, ch)`` — feature and channel dimensions.

        Returns
        -------
        SMPath
            The configured score matching path.
        """
        path = build_sm_path(self.sde_type, config)

        if self.sde_type == "VP":
            self.prior = make_gaussian_prior(*event_shape)
        else:  # VE
            self.prior = make_gaussian_prior(*event_shape, sigma=path.scheduler.sigma_max)

        return path

    def build_loss(self, path, weights=None):
        """Build the score matching loss.

        Wraps ``path.get_loss_fn()`` into a callable object.

        Parameters
        ----------
        path : SMPath
            The score matching path.
        weights : Array, optional
            Per-dimension loss weights.

        Returns
        -------
        SMLoss
            A loss callable with signature
            ``(key, model, batch, condition_mask=None, model_extras=None) -> loss``.
        """
        return SMLoss(path, weights=weights)

    def prepare_batch(self, key, x_1, path):
        """Sample noise and diffusion time for a score matching training batch.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_1 : Array
            Clean data of shape ``(batch_size, dim, ch)``.
        path : SMPath
            The score matching path.

        Returns
        -------
        tuple
            ``(x_0, x_1, t)`` where ``x_0`` is standard normal noise
            and ``t`` has shape ``(batch_size, 1, 1)``.
        """
        rng_x0, rng_t = jax.random.split(key)
        x_0 = jax.random.normal(rng_x0, x_1.shape)
        t = path.sample_t(rng_t, (x_1.shape[0], 1, 1))
        return (x_0, x_1, t)

    def get_default_solver(self):
        """Return the default reverse SDE solver.

        Returns
        -------
        tuple
            ``(NewSMSDESolver, {})``
        """
        return (NewSMSDESolver, {})

    def build_solver(self, model_wrapped, path, solver=None):
        """Instantiate a score matching solver.

        For the reverse SDE (``SMSDESolver``), wraps the model and extracts
        prior parameters. For the probability flow ODE (``SMODESolver``),
        wraps the score model with ``ScoreToODEDrift`` first.

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : SMPath
            The score matching path.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(NewSMSDESolver, {})``.

        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver

        if issubclass(solver_cls, NewSMODESolver):
            # PF-ODE path: wrap score model as drift model
            drift_model = ScoreToODEDrift(
                score_model=model_wrapped, sde=path.scheduler
            )
            wrapper = ModelWrapper(model=drift_model)
            return solver_cls(velocity_model=wrapper, **solver_kwargs)
        else:
            # SDE path (NewSMSDESolver): wrap model, pass SDE scheduler
            wrapper = ModelWrapper(model=model_wrapped)
            sde = path.scheduler
            return solver_cls(
                velocity_model=wrapper,
                sde=sde,
                **solver_kwargs,
            )

    def sample_init(self, key, nsamples):
        """Sample from the score matching prior.

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

    def build_sampler_fn(
        self,
        model_wrapped,
        path,
        model_extras,
        nsteps=1000,
        method="Euler",
        return_intermediates=False,
        solver=None,
        **kwargs,
    ):
        """Build a sampler closure for score matching.

        Supports ``SMSDESolver`` (reverse SDE) and ``SMODESolver``
        (probability flow ODE via ``ScoreToODEDrift``).

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : SMPath
            The score matching path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, etc.).
        nsteps : int, optional
            Number of integration steps. Default is 1000.
        method : str or diffrax solver, optional
            Integration method. Default is ``"Euler"``.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(NewSMSDESolver, {})``.

        Returns
        -------
        sampler_fn : Callable
            A function ``(key, x_init) -> samples``.
        """
        sde = path.scheduler
        T = sde.T
        eps = 1e-3

        if return_intermediates:
            time_grid = jnp.linspace(T, eps, nsteps + 1)
        else:
            time_grid = jnp.array([T, eps])

        step_size = -(T - eps) / nsteps

        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver

        if issubclass(solver_cls, NewSMODESolver):
            # PF-ODE path (deterministic) — build solver eagerly
            solver_instance = self.build_solver(model_wrapped, path, solver=(solver_cls, solver_kwargs))
            sampler_ = solver_instance.get_sampler(
                step_size=step_size,
                method=method,
                time_grid=time_grid,
                return_intermediates=return_intermediates,
            )

            def sampler_fn(key, x_init, model_extras=None):
                if model_extras is None:
                    model_extras = {}
                return sampler_(x_init, model_extras=model_extras)
        elif issubclass(solver_cls, NewSMSDESolver):
            # Reverse SDE path (stochastic) — build solver eagerly
            solver_instance = self.build_solver(model_wrapped, path, solver=(solver_cls, solver_kwargs))
            sampler_ = solver_instance.get_sampler(
                step_size=step_size,
                method=method,
                time_grid=time_grid,
                return_intermediates=return_intermediates,
            )

            def sampler_fn(key, x_init, model_extras=None):
                if model_extras is None:
                    model_extras = {}
                return sampler_(x_init, key=key, model_extras=model_extras)
        else:
            raise ValueError(f"Unsupported solver type: {solver_cls}")

        return sampler_fn

    def build_log_prob_fn(
        self,
        model_wrapped,
        path,
        model_extras,
        step_size=0.01,
        method="Dopri5",
        atol=1e-5,
        rtol=1e-5,
        nsteps=1000,
        solver=None,
        exact_divergence=True,
        log_prior=None,
        **kwargs,
    ):
        """Build a log-probability closure for the score matching probability flow ODE.

        Uses the continuous change-of-variables formula via ``SMODESolver``
        (which inherits ``get_log_prob`` from ``ODESolver``).  Only works
        with ``SMODESolver``; passing ``SMSDESolver`` will raise an error.

        .. note::

           Unlike the default sampling behaviour of ``ScoreMatchingMethod``,
           which integrates the *reverse SDE* (via ``SMSDESolver``),
           log-probability evaluation requires the *probability flow ODE*
           formulation (via ``SMODESolver``).  Both formulations share the
           same marginal distributions, so a score model trained with the
           standard SM loss is mathematically valid for the probability
           flow ODE.  When no ``solver`` is passed, ``SMODESolver`` is
           selected automatically.

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : SMPath
            The score matching path.
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
        nsteps : int, optional
            Number of integration steps (used for step_size calculation
            when a fixed-step solver is selected). Default is 1000.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Must be an ODE-based solver
            (``SMODESolver``).
        exact_divergence : bool, optional
            If ``True`` (default), compute exact divergence via full
            Jacobian. If ``False``, use the Hutchinson estimator (requires
            a PRNG ``key`` at call time).
        log_prior : callable, optional
            Override for the prior's ``log_prob``. If ``None``, uses
            ``self.prior.log_prob``. Used by the joint pipeline.

        Returns
        -------
        log_prob_fn : Callable
            ``(x_1, model_extras=None, *, key=None) -> log_prob``.

        Raises
        ------
        NotImplementedError
            If a non-ODE solver (e.g. ``SMSDESolver``) is specified.
        """
        if solver is None:
            solver = (NewSMODESolver, {})

        solver_instance = self.build_solver(model_wrapped, path, solver=solver)

        if not isinstance(solver_instance, NewSMODESolver):
            raise NotImplementedError(
                f"Log-probability computation requires SMODESolver, "
                f"got {type(solver_instance).__name__}."
            )

        # SM time grid: T (noise) → eps (near-data)
        sde = path.scheduler
        T = sde.T
        eps = 1e-3
        time_grid = jnp.array([T, eps])

        # Step size: positive — ODESolver.get_log_prob already negates it
        step_size = (T - eps) / nsteps

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

    def get_extra_training_config(self):
        """Return SM-specific training config defaults.

        Returns
        -------
        dict
            Scheduler defaults for the selected SDE type.
        """
        if self.sde_type == "VP":
            return {"beta_min": 0.001, "beta_max": 3.0}
        elif self.sde_type == "VE":
            return {"sigma_min": 0.0001, "sigma_max": 15.0}
        return {}
