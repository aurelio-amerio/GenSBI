from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array

from gensbi.solver import Solver
from gensbi.diffusion.solver.edm_samplers import edm_sampler, edm_ablation_sampler
from gensbi.diffusion.path import EDMPath


class EDMSolver(Solver):
    def __init__(self, score_model: Callable, path: EDMPath) -> None:
        """
        Initialize the SDE solver.

        Parameters
        ----------
            score_model : Callable
                The score model function.
            path : EDMPath
                The EDMPath object.

        Example:
            .. code-block:: python

                from gensbi.diffusion.solver import EDMSolver
                from gensbi.diffusion.path import EDMPath
                from gensbi.diffusion.path.scheduler import EDMScheduler
                import jax, jax.numpy as jnp
                scheduler = EDMScheduler()
                path = EDMPath(scheduler)
                def score_model(x, t):
                    return x + t
                solver = EDMSolver(score_model, path)
                key = jax.random.PRNGKey(0)
                x_init = jax.random.normal(key, (16, 2))
                samples = solver.sample(key, x_init, nsteps=10)
                print(samples.shape)
                # (10, 16, 2)

        """
        self.score_model = score_model
        self.path = path
        assert self.path.scheduler.name in [
            "EDM",
            "EDM-VP",
            "EDM-VE",
        ], f"Path must be one of ['EDM', 'EDM-VP', 'EDM-VE'], got {self.path.name}."

    def get_sampler(
        self,
        cfg_scale: Optional[float] = None,
        nsteps: int = 18,
        method: str = "Heun",
        return_intermediates: bool = False,
        static_model_kwargs: dict = None,
        solver_params: Optional[dict] = None,
        solver_scheduler: Optional[Any] = None,
    ) -> Callable:
        """
        Returns a sampler function for the SDE.

        Parameters
        ----------
            cfg_scale : Optional[float]
                Classifier-free guidance scale (not implemented).
            nsteps : int
                Number of steps.
            method : str
                Integration method.
            return_intermediates : bool
                Whether to return intermediate steps.
            static_model_kwargs : dict
                Static model arguments baked into the sampler.
                Condition-dependent data should be passed at call time
                via ``model_extras``.
            solver_params : Optional[dict]
                Additional solver parameters.
            solver_scheduler : Optional[Any]
                Scheduler to use for the solver. If None, the path's scheduler is used.

        Returns
        -------
            Callable
                ``sample(key, x_init, model_extras=None)`` sampler function.
        """
        if solver_scheduler is None:
            solver_scheduler = self.path.scheduler

        if static_model_kwargs is None:
            static_model_kwargs = {}
        if solver_params is None:
            solver_params = {}

        if solver_scheduler.name == "EDM":
            sampler_ = edm_sampler
        else:
            # Bind the training scheduler as denoise_scheduler so the model
            # is always called with the preconditioning it was trained with.
            # Use a lambda (not partial) to insert denoise_scheduler in the
            # correct positional slot while keeping the same call signature
            # as edm_sampler: (sched, model, x_1, **kw).
            _denoise_sched = self.path.scheduler
            sampler_ = lambda sched, model, x_1, **kw: edm_ablation_sampler(
                sched, _denoise_sched, model, x_1, **kw
            )

        if cfg_scale is not None:
            raise NotImplementedError(
                "CFG scale is not implemented for EDM samplers yet."
            )

        S_churn = solver_params.get("S_churn", 0)  # type: ignore
        S_min = solver_params.get("S_min", 0)  # type: ignore
        S_max = solver_params.get("S_max", float("inf"))  # type: ignore
        S_noise = solver_params.get("S_noise", 1)  # type: ignore

        @jit
        def sample(key: Array, x_init: Array, model_extras=None) -> Array:
            if model_extras is None:
                model_extras = {}
            return sampler_(
                solver_scheduler,
                self.score_model,
                x_init,
                key=key,
                return_intermediates=return_intermediates,
                n_steps=nsteps,
                S_churn=S_churn,
                S_min=S_min,
                S_max=S_max,
                S_noise=S_noise,
                method=method,
                model_kwargs={**static_model_kwargs, **model_extras},
            )

        return sample

    def sample(
        self,
        key: Array,
        x_init: Array,
        cfg_scale: Optional[float] = None,
        nsteps: int = 18,
        method: str = "Heun",
        return_intermediates: bool = False,
        model_extras: dict = None,
        solver_params: Optional[dict] = None,
        solver_scheduler: Optional[Any] = None,
    ) -> Array:
        """
        Sample from the SDE using the sampler.

        Parameters
        ----------
            key : Array
                JAX random key.
            x_init : Array
                Initial value.
            cfg_scale : Optional[float]
                Classifier-free guidance scale (not implemented).
            nsteps : int
                Number of steps.
            method : str
                Integration method.
            return_intermediates : bool
                Whether to return intermediate steps.
            model_extras : dict
                Runtime model extras (e.g. ``cond``, ``obs_ids``).
            solver_params : Optional[dict]
                Additional solver parameters.
            solver_scheduler : Optional[Any]
                Scheduler to use for the solver. If None, the path's scheduler is used.

        Returns
        -------
            Array
                Sampled output.
        """
        sample = self.get_sampler(
            cfg_scale=cfg_scale,
            nsteps=nsteps,
            method=method,
            return_intermediates=return_intermediates,
            solver_params=solver_params,
            solver_scheduler=solver_scheduler,
        )
        return sample(key, x_init, model_extras=model_extras)
