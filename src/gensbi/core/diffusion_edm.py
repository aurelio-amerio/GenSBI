"""
EDM diffusion generative method strategy.

Implements :class:`~gensbi.core.generative_method.GenerativeMethod` using
Elucidated Diffusion Models (EDM) with support for EDM, VP, and VE
SDE formulations.
"""

import jax
import jax.numpy as jnp

from gensbi.core.generative_method import GenerativeMethod
from gensbi.recipes.utils import build_edm_path
from gensbi.diffusion.solver import EDMSolver


class DiffusionEDMMethod(GenerativeMethod):
    """EDM diffusion strategy.

    Supports three SDE formulations via the ``sde`` parameter:

    - ``"EDM"`` — standard EDM schedule (default)
    - ``"VP"`` — variance-preserving schedule
    - ``"VE"`` — variance-exploding schedule

    Parameters
    ----------
    sde : str, optional
        SDE type. One of ``"EDM"``, ``"VP"``, ``"VE"``. Default is ``"EDM"``.

    Examples
    --------
    >>> method = DiffusionEDMMethod(sde="EDM")
    >>> path = method.build_path(config={"sigma_min": 0.002, "sigma_max": 80.0})
    >>> loss = method.build_loss(path)
    """

    def __init__(self, sde="EDM"):
        if sde not in ("EDM", "VP", "VE"):
            raise ValueError(
                f"sde must be one of 'EDM', 'VP', 'VE', got '{sde}'."
            )
        self.sde = sde

    def build_path(self, config):
        """Build an EDM diffusion path.

        Parameters
        ----------
        config : dict
            Training configuration. Reads scheduler hyperparameters
            (``sigma_min``, ``sigma_max``, ``beta_min``, ``beta_max``)
            with sensible defaults.

        Returns
        -------
        EDMPath
            The configured diffusion path.
        """
        return build_edm_path(self.sde, config)

    def build_loss(self, path):
        """Build the EDM denoising loss.

        Wraps ``path.get_loss_fn()`` into a callable object.

        Parameters
        ----------
        path : EDMPath
            The diffusion path.

        Returns
        -------
        EDMLoss
            A loss callable with signature
            ``(key, model, batch, condition_mask=None, model_extras={}) -> loss``.
        """
        return EDMLoss(path)

    def prepare_batch(self, key, x_1, path):
        """Sample noise and sigma for an EDM training batch.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_1 : Array
            Clean data of shape ``(batch_size, dim, ch)``.
        path : EDMPath
            The diffusion path.

        Returns
        -------
        tuple
            ``(x_0, x_1, sigma)`` where ``x_0`` is standard normal noise
            and ``sigma`` has shape ``(batch_size, 1, 1)``.
        """
        rng_x0, rng_sigma = jax.random.split(key)
        x_0 = jax.random.normal(rng_x0, x_1.shape)
        sigma = path.sample_sigma(rng_sigma, (x_1.shape[0], 1, 1))
        return (x_0, x_1, sigma)

    def get_default_solver(self):
        """Return the default EDM solver.

        Returns
        -------
        tuple
            ``(EDMSolver, {})``
        """
        return (EDMSolver, {})

    def build_solver(self, model_wrapped, path, solver=None):
        """Instantiate an EDM solver.

        The solver class must accept ``score_model`` and ``path`` as
        its first two positional arguments.

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : EDMPath
            The diffusion path.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(EDMSolver, {})``.

        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        if solver is None:
            solver = self.get_default_solver()
        solver_cls, solver_kwargs = solver
        return solver_cls(score_model=model_wrapped, path=path, **solver_kwargs)

    def sample_init(self, key, shape, path):
        """Sample from the diffusion prior.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        shape : tuple
            Shape of the initial sample.
        path : EDMPath
            The diffusion path (provides ``sample_prior``).

        Returns
        -------
        Array
            Sample from the prior.
        """
        return path.sample_prior(key, shape)

    def build_sampler_fn(self, model_wrapped, path, model_extras,
                         nsteps=18, return_intermediates=False,
                         solver=None, solver_scheduler=None,
                         solver_params=None, **kwargs):
        """Build a sampler closure for EDM diffusion.

        Supports ``EDMSolver`` with EDM, VP, and VE schedulers.
        The ``solver_scheduler`` can override the path's scheduler for
        sampling (also used for ``sample_prior``).

        Parameters
        ----------
        model_wrapped
            The wrapped score model.
        path : EDMPath
            The diffusion path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, etc.).
        nsteps : int, optional
            Number of sampling steps. Default is 18.
        return_intermediates : bool, optional
            Whether to return intermediate steps. Default is False.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)``. Defaults to ``(EDMSolver, {})``.
        solver_scheduler : optional
            Override scheduler for the solver. If ``None``, uses the
            path's scheduler.
        solver_params : dict, optional
            EDM solver parameters (``S_churn``, ``S_min``, ``S_max``,
            ``S_noise``).

        Returns
        -------
        sampler_fn : Callable
            A function ``(key, x_init) -> samples``.
        """
        if solver_params is None:
            solver_params = {}

        solver_instance = self.build_solver(model_wrapped, path, solver=solver)

        sampler_ = solver_instance.get_sampler(
            nsteps=nsteps,
            return_intermediates=return_intermediates,
            solver_scheduler=solver_scheduler,
            solver_params=solver_params,
        )

        def sampler_fn(key, x_init, model_extras={}):
            return sampler_(key, x_init, model_extras=model_extras)

        return sampler_fn

    def get_extra_training_config(self):
        """Return EDM-specific training config defaults.

        Returns
        -------
        dict
            Scheduler defaults for the selected SDE type.
        """
        if self.sde == "EDM":
            return {"sigma_min": 0.002, "sigma_max": 80.0}
        elif self.sde == "VE":
            return {"sigma_min": 0.02, "sigma_max": 100.0}
        elif self.sde == "VP":
            return {"beta_min": 0.1, "beta_max": 20.0}
        return {}


from gensbi.diffusion.loss import EDMLoss  # noqa: E402
