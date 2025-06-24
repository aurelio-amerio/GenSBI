from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array


from gensbi.diffusion.solver.solver import Solver
from gensbi.diffusion.solver.edm_samplers import edm_sampler, edm_ablation_sampler
from gensbi.diffusion.path import EDMPath


class SDESolver(Solver):
    def __init__(self, score_model: Callable, path: EDMPath):

        self.score_model = score_model
        self.path = path
        assert self.path.scheduler.name in [
            "EDM",
            "EDM-VP",
            "EDM-VE",
        ], f"Path must be one of ['EDM', 'EDM-VP', 'EDM-VE'], got {self.path.name}."


    def get_sampler(
        self,
        condition_mask: Optional[Array] = None,
        condition_value: Optional[Array] = None,
        cfg_scale=None,
        nsteps=18,
        method: str = "Heun",
        return_intermediates: bool = False,
        model_extras: dict = {},
        solver_params: Optional[dict] = {},
    ):
        if self.path.name == "EDM":
            sampler_ = edm_sampler
        else:
            sampler_ = edm_ablation_sampler

        if cfg_scale is not None:
            raise NotImplementedError(
                "CFG scale is not implemented for EDM samplers yet."
            )

        # if return_intermediates:
        #     raise NotImplementedError(
        #         "Returning intermediates is not implemented for EDM samplers yet."
        #     )

        # wrap the sampler
        S_churn = solver_params.get("S_churn", 0) # type: ignore
        S_min = solver_params.get("S_min", 0) # type: ignore
        S_max = solver_params.get("S_max", float("inf")) # type: ignore
        S_noise = solver_params.get("S_noise", 1) # type: ignore

        @jit
        def sample(key, x_init):
            return sampler_(
                self.path.scheduler,
                self.score_model,
                x_init,
                key=key,
                condition_mask=condition_mask,
                condition_value=condition_value,
                return_intermediates=return_intermediates,
                n_steps=nsteps,
                S_churn=S_churn,
                S_min=S_min,
                S_max=S_max,
                S_noise=S_noise,
                method=method,
                model_kwargs=model_extras,
            )

        return sample

    def sample(
        self,
        key,
        x_init: Array,
        condition_mask: Optional[Array] = None,
        condition_value: Optional[Array] = None,
        cfg_scale=None,
        nsteps=18,
        method: str = "Heun",
        return_intermediates: bool = False,
        model_extras: dict = {},
        solver_params: Optional[dict] = {},
    ):
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
