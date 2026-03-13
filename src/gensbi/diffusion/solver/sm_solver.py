"""
Score Matching Solvers.

This module provides solvers for score matching diffusion models:
``SMSolver`` for reverse SDE sampling and ``SMPFSolver`` (a thin
``ODESolver`` subclass) for probability flow ODE sampling.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array

from gensbi.solver import Solver
from gensbi.diffusion.solver.sm_samplers import sm_reverse_sde_sampler
from gensbi.diffusion.path.sm_path import SMPath


class SMSolver(Solver):
    """
    Solver for standard score matching diffusion models.

    Uses reverse SDE sampling via Euler-Maruyama integration.

    Parameters
    ----------
        score_model : Callable
            The score model function.
        path : SMPath
            The SMPath object containing the SDE scheduler.

    Example:
        .. code-block:: python

            from gensbi.diffusion.solver.sm_solver import SMSolver
            from gensbi.diffusion.path.sm_path import SMPath
            from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler
            import jax, jax.numpy as jnp
            sde = VPSmScheduler()
            path = SMPath(sde)
            def score_model(obs, t, **kwargs):
                return jnp.zeros_like(obs)
            solver = SMSolver(score_model, path)
            key = jax.random.PRNGKey(0)
            x_init = jax.random.normal(key, (16, 2))
            samples = solver.sample(key, x_init, nsteps=10)
            print(samples.shape)
            # (16, 2)
    """

    def __init__(self, score_model: Callable, path: SMPath) -> None:
        import warnings
        warnings.warn(
            "SMSolver is deprecated, use SMSDESolver (from gensbi.diffusion.solver) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.score_model = score_model
        self.path = path
        assert self.path.name in [
            "SM-VP",
            "SM-VE",
        ], f"Path must be one of ['SM-VP', 'SM-VE'], got {self.path.name}."

    def get_sampler(
        self,
        condition_mask: Optional[Array] = None,
        condition_value: Optional[Array] = None,
        cfg_scale: Optional[float] = None,
        nsteps: int = 1000,
        method: str = "Euler",
        return_intermediates: bool = False,
        static_model_kwargs: dict = None,
        solver_params: Optional[dict] = None,
    ) -> Callable:
        """
        Returns a sampler function for the reverse SDE.

        Parameters
        ----------
            condition_mask : Optional[Array]
                Mask for conditioning.
            condition_value : Optional[Array]
                Value for conditioning.
            cfg_scale : Optional[float]
                Classifier-free guidance scale (not implemented).
            nsteps : int
                Number of integration steps.
            method : str
                Integration method. One of "Euler", "Heun", "SEA", "ShARK".
            return_intermediates : bool
                Whether to return intermediate steps.
            static_model_kwargs : dict
                Static model arguments baked into the sampler.
                Condition-dependent data should be passed at call time
                via ``model_extras``.
            solver_params : Optional[dict]
                Additional solver parameters.

        Returns
        -------
            Callable
                ``sample(key, x_init, model_extras=None)`` sampler function.
        """
        if cfg_scale is not None:
            raise NotImplementedError(
                "CFG scale is not implemented for SM samplers yet."
            )

        if static_model_kwargs is None:
            static_model_kwargs = {}
        if solver_params is None:
            solver_params = {}

        eps = solver_params.get("eps", 1e-3)  # type: ignore

        @jit
        def sample(key: Array, x_init: Array, model_extras=None) -> Array:
            if model_extras is None:
                model_extras = {}
            return sm_reverse_sde_sampler(
                self.path.scheduler,
                self.score_model,
                x_init,
                key=key,
                condition_mask=condition_mask,
                condition_value=condition_value,
                return_intermediates=return_intermediates,
                n_steps=nsteps,
                eps=eps,
                method=method,
                model_kwargs={**static_model_kwargs, **model_extras},
            )

        return sample

    def sample(
        self,
        key: Array,
        x_init: Array,
        condition_mask: Optional[Array] = None,
        condition_value: Optional[Array] = None,
        cfg_scale: Optional[float] = None,
        nsteps: int = 1000,
        method: str = "Euler",
        return_intermediates: bool = False,
        model_extras: dict = None,
        solver_params: Optional[dict] = None,
    ) -> Array:
        """
        Sample from the reverse SDE.

        Parameters
        ----------
            key : Array
                JAX random key.
            x_init : Array
                Initial value from prior.
            condition_mask : Optional[Array]
                Mask for conditioning.
            condition_value : Optional[Array]
                Value for conditioning.
            cfg_scale : Optional[float]
                Classifier-free guidance scale (not implemented).
            nsteps : int
                Number of integration steps.
            method : str
                Integration method. One of "Euler", "Heun", "SEA", "ShARK".
            return_intermediates : bool
                Whether to return intermediate steps.
            model_extras : dict
                Additional model arguments.
            solver_params : Optional[dict]
                Additional solver parameters.

        Returns
        -------
            Array
                Sampled output.
        """
        sample = self.get_sampler(
            condition_mask=condition_mask,
            condition_value=condition_value,
            cfg_scale=cfg_scale,
            nsteps=nsteps,
            method=method,
            return_intermediates=return_intermediates,
            solver_params=solver_params,
        )
        return sample(key, x_init, model_extras=model_extras)


from gensbi.flow_matching.solver import ODESolver


class SMPFSolver(ODESolver):
    r"""Score matching PF-ODE solver (probability flow ODE).

    Thin subclass of :class:`ODESolver` used for dispatching in the
    score matching pipeline.  ``ScoreMatchingMethod.build_solver``
    detects this type and wraps the score model with
    :class:`~gensbi.utils.model_wrapping.ScoreToODEDrift` before
    constructing the solver.

    The PF-ODE is:

    .. math::
        dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x, t) \right] dt

    All integration logic is inherited from ``ODESolver``.

    See Also
    --------
    gensbi.utils.model_wrapping.ScoreToODEDrift
    gensbi.core.score_matching.ScoreMatchingMethod.build_solver
    """

    def __init__(self, velocity_model):
        import warnings
        warnings.warn(
            "SMPFSolver is deprecated, use SMODESolver (from gensbi.diffusion.solver) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(velocity_model=velocity_model)
