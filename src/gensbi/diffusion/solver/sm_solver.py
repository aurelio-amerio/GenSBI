"""
Score Matching SDE Solver.

This module provides a solver for score matching diffusion models,
handling reverse SDE sampling using the learned score function.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array

from gensbi.solver import Solver
from gensbi.diffusion.solver.sm_samplers import (
    sm_reverse_sde_sampler,
    sm_reverse_ode_sampler,
)
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
        model_extras: dict = {},
        solver_params: Optional[dict] = {},
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
            model_extras : dict
                Additional model arguments.
            solver_params : Optional[dict]
                Additional solver parameters.

        Returns
        -------
            Callable
                Sampler function.
        """
        if cfg_scale is not None:
            raise NotImplementedError(
                "CFG scale is not implemented for SM samplers yet."
            )

        eps = solver_params.get("eps", 1e-3)  # type: ignore

        @jit
        def sample(key: Array, x_init: Array, model_extras=model_extras) -> Array:
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
                model_kwargs=model_extras,
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
        model_extras: dict = {},
        solver_params: Optional[dict] = {},
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
            model_extras=model_extras,
            solver_params=solver_params,
        )
        return sample(key, x_init)


class SMPFSolver(SMSolver):
    r"""
    Solver for score matching diffusion models using the Probability Flow ODE.

    Instead of the reverse SDE, this solver integrates the deterministic
    Probability Flow ODE (PF-ODE) from section 4.3 of Song et al., 2021
    (arXiv:2011.13456). The PF-ODE shares the same marginal distributions
    as the reverse SDE but is fully deterministic, which often leads to
    higher sample quality and enables exact likelihood computation.

    The PF-ODE is:

    .. math::
        dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x, t) \right] dt

    Parameters
    ----------
        score_model : Callable
            The score model function.
        path : SMPath
            The SMPath object containing the SDE scheduler.

    Example:
        .. code-block:: python

            from gensbi.diffusion.solver.sm_solver import SMPFSolver
            from gensbi.diffusion.path.sm_path import SMPath
            from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler
            import jax, jax.numpy as jnp
            sde = VPSmScheduler()
            path = SMPath(sde)
            def score_model(obs, t, **kwargs):
                return jnp.zeros_like(obs)
            solver = SMPFSolver(score_model, path)
            key = jax.random.PRNGKey(0)
            x_init = jax.random.normal(key, (16, 2, 1))
            samples = solver.sample(key, x_init, nsteps=10)
            print(samples.shape)
            # (16, 2, 1)
    """

    def __init__(self, score_model: Callable, path: SMPath) -> None:
        super().__init__(score_model, path)

    def get_sampler(
        self,
        condition_mask: Optional[Array] = None,
        condition_value: Optional[Array] = None,
        cfg_scale: Optional[float] = None,
        nsteps: int = 1000,
        method: str = "Euler",
        return_intermediates: bool = False,
        model_extras: dict = {},
        solver_params: Optional[dict] = {},
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> Callable:
        """
        Returns a sampler function for the probability flow ODE (PF-ODE).

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
                Integration method. One of ``"Euler"``, ``"Heun"``, ``"Dopri5"``.
                ``"Dopri5"`` automatically uses adaptive step sizing via
                ``PIDController``; the others use fixed step size.
            return_intermediates : bool
                Whether to return intermediate steps.
            model_extras : dict
                Additional model arguments.
            solver_params : Optional[dict]
                Additional solver parameters.
            atol : float
                Absolute tolerance for adaptive step solvers.
            rtol : float
                Relative tolerance for adaptive step solvers.

        Returns
        -------
            Callable
                Sampler function.
        """
        if cfg_scale is not None:
            raise NotImplementedError(
                "CFG scale is not implemented for SM samplers yet."
            )

        eps = solver_params.get("eps", 1e-3)  # type: ignore

        @jit
        def sample(key: Array, x_init: Array, model_extras=model_extras) -> Array:
            return sm_reverse_ode_sampler(
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
                atol=atol,
                rtol=rtol,
                model_kwargs=model_extras,
            )

        return sample
