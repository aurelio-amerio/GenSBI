"""
Schedule transformation utilities for flow matching.

This module provides functionality to transform the time scheduling of a velocity model
from one scheduler to another, allowing for post-training schedule modifications.
"""
from jax import Array

from gensbi.flow_matching.path.scheduler.scheduler import Scheduler
from gensbi.utils.model_wrapping import ModelWrapper

from flax import nnx


class ScheduleTransformedModel(ModelWrapper):
    r"""
    Change of scheduler for a velocity model.

    This class wraps a given velocity model and transforms its scheduling
    to a new scheduler function. It modifies the time
    dynamics of the model according to the new scheduler while maintaining
    the original model's behavior.

    Example:

        .. code-block:: python

            import jax
            import jax.numpy as jnp
            from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, ScheduleTransformedModel
            from flow_matching.solver import ODESolver

            # Initialize the model and schedulers
            model = ...

            original_scheduler = CondOTScheduler()
            new_scheduler = CosineScheduler()

            # Create the transformed model
            transformed_model = ScheduleTransformedModel(
                velocity_model=model,
                original_scheduler=original_scheduler,
                new_scheduler=new_scheduler
            )

            # Set up the solver
            solver = ODESolver(velocity_model=transformed_model)

            key = jax.random.PRNGKey(0)
            x_0 = jax.random.normal(key, shape=(10, 2))  # Example initial condition

            x_1 = solver.sample(
                time_steps=jnp.array([0.0, 1.0]),
                x_init=x_0,
                step_size=1/1000
                )[1]

    Parameters
    ----------
        velocity_model : ModelWrapper
            The original velocity model to be transformed.
        original_scheduler : Scheduler
            The scheduler used by the original model. Must implement the snr_inverse function.
        new_scheduler : Scheduler
            The new scheduler to be applied to the model.
    """

    def __init__(
        self,
        velocity_model: nnx.Module,
        original_scheduler: Scheduler,
        new_scheduler: Scheduler,
    ) -> None:
        """
        Initialize the ScheduleTransformedModel.

        Parameters
        ----------
            velocity_model : nnx.Module
                The original velocity model.
            original_scheduler : Scheduler
                The scheduler used by the original model.
            new_scheduler : Scheduler
                The new scheduler to be applied.
        """
        super().__init__(model=velocity_model)
        self.original_scheduler = original_scheduler
        self.new_scheduler = new_scheduler

        assert hasattr(self.original_scheduler, "snr_inverse") and callable(
            getattr(self.original_scheduler, "snr_inverse")
        ), "The original scheduler must have a callable 'snr_inverse' method."

    def __call__(self, x: Array, t: Array, **extras) -> Array:
        r"""
        Compute the transformed marginal velocity field for a new scheduler.
        This method implements a post-training velocity scheduler change for
        affine conditional flows.

        Parameters
        ----------
            x : Array
                :math:`x_t`, the input array.
            t : Array
                The time array (denoted as :math:`r` above).
            **extras: Additional arguments for the model.

        Returns
        -------
            Array
                The transformed velocity.
        """
        r = t

        r_scheduler_output = self.new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        d_alpha_r = r_scheduler_output.d_alpha_t
        d_sigma_r = r_scheduler_output.d_sigma_t

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)

        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = t_scheduler_output.alpha_t
        sigma_t = t_scheduler_output.sigma_t
        d_alpha_t = t_scheduler_output.d_alpha_t
        d_sigma_t = t_scheduler_output.d_sigma_t

        s_r = sigma_r / sigma_t

        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        u_t = self.model(x=x / s_r, t=t, **extras)  # type: ignore
        u_r = ds_r * x / s_r + dt_r * s_r * u_t

        return u_r
