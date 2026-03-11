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
from gensbi.diffusion.solver import SMSolver, SMPFSolver

from gensbi.diffusion.loss import SMLoss
from gensbi.diffusion.sm_prior import VPPrior, VEPrior
from gensbi.flow_matching.solver import ODESolver
from gensbi.utils.model_wrapping import ModelWrapper, ScoreToDrift


class ScoreMatchingMethod(GenerativeMethod):
    """Score matching strategy.

    Supports two SDE formulations via the ``sde_type`` parameter:

    - ``"VP"`` — variance-preserving (default)
    - ``"VE"`` — variance-exploding

    Sampling can use either the reverse SDE (``SMSolver``, default) or
    the probability flow ODE (``SMPFSolver`` via ``ScoreToDrift``).

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

        # Prior distribution (matches FM's StandardNormalPrior pattern)
        if sde_type == "VP":
            self.prior = VPPrior()
        else:
            # VEPrior requires sigma_max, which depends on the path config.
            # Initialized to None; set properly when build_path is called.
            self.prior = None

    def build_path(self, config):
        """Build a score matching path.

        Parameters
        ----------
        config : dict
            Training configuration. Reads scheduler hyperparameters
            (``beta_min``, ``beta_max`` for VP; ``sigma_min``, ``sigma_max``
            for VE) with sensible defaults.

        Returns
        -------
        SMPath
            The configured score matching path.
        """
        path = build_sm_path(self.sde_type, config)

        # Set VE prior now that we know sigma_max
        if self.sde_type == "VE" and self.prior is None:
            self.prior = VEPrior(sigma_max=path.scheduler.sigma_max)

        return path

    def build_loss(self, path):
        """Build the score matching loss.

        Wraps ``path.get_loss_fn()`` into a callable object.

        Parameters
        ----------
        path : SMPath
            The score matching path.

        Returns
        -------
        SMLoss
            A loss callable with signature
            ``(key, model, batch, condition_mask=None, model_extras=None) -> loss``.
        """
        return SMLoss(path)

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
            ``(SMSolver, {})``
        """
        return (SMSolver, {})

    def build_solver(self, model_wrapped, path, solver=None):
        """Instantiate a score matching solver.

        For the reverse SDE (``SMSolver``), instantiates the solver
        directly. For PF-ODE sampling (``SMPFSolver``), wraps the score model with
        ``ScoreToDrift`` first.

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : SMPath
            The score matching path.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(SMSolver, {})``.

        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver

        if issubclass(solver_cls, SMPFSolver):
            # PF-ODE path: wrap score model as drift model
            drift_model = ScoreToDrift(
                score_model=model_wrapped, sde=path.scheduler
            )
            wrapper = ModelWrapper(model=drift_model)
            return solver_cls(velocity_model=wrapper, **solver_kwargs)
        else:
            # SDE path (SMSolver): pass directly
            return solver_cls(
                score_model=model_wrapped, path=path, **solver_kwargs
            )

    def sample_init(self, key, shape, path):
        """Sample from the score matching prior.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        shape : tuple
            Shape of the initial sample.
        path : SMPath
            The score matching path (provides ``sample_prior``).

        Returns
        -------
        Array
            Sample from the prior.
        """
        return path.sample_prior(key, shape)

    def build_sampler_fn(
        self,
        model_wrapped,
        path,
        model_extras,
        nsteps=1000,
        return_intermediates=False,
        solver=None,
        **kwargs,
    ):
        """Build a sampler closure for score matching.

        Supports ``SMSolver`` (reverse SDE) and ``SMPFSolver``
        (PF-ODE via ``ScoreToDrift`` + ``ODESolver``).

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
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(SMSolver, {})``.

        Returns
        -------
        sampler_fn : Callable
            A function ``(key, x_init) -> samples``.
        """
        solver_instance = self.build_solver(model_wrapped, path, solver=solver)

        if isinstance(solver_instance, SMPFSolver):
            # PF-ODE via SMPFSolver: compute SM-specific time grid and step size
            sde = path.scheduler
            T = sde.T
            eps = 1e-3

            if return_intermediates:
                time_grid = jnp.linspace(T, eps, nsteps + 1)
            else:
                time_grid = jnp.array([T, eps])

            step_size = -(T - eps) / nsteps

            sampler_ = solver_instance.get_sampler(
                step_size=step_size,
                method="Euler",
                time_grid=time_grid,
                return_intermediates=return_intermediates,
            )

            def sampler_fn(key, x_init, model_extras=None):
                if model_extras is None:
                    model_extras = {}
                return sampler_(x_init, model_extras=model_extras)
        elif isinstance(solver_instance, SMSolver):
            # SDE path (SMSolver)
            sampler_ = solver_instance.get_sampler(
                nsteps=nsteps,
                return_intermediates=return_intermediates,
            )

            def sampler_fn(key, x_init, model_extras=None):
                if model_extras is None:
                    model_extras = {}
                return sampler_(key, x_init, model_extras=model_extras)
        else:
            raise ValueError(f"Unsupported solver type: {type(solver_instance)}")

        return sampler_fn

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
