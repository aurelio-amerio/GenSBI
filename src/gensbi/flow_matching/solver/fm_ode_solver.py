"""
Flow matching ODE solver.

Provides :class:`FMODESolver`, where the drift is simply the
velocity field from the wrapped model.
"""

from typing import Callable

from gensbi.core.ode_solver import ODESolver
from gensbi.utils.model_wrapping import ModelWrapper


class FMODESolver(ODESolver):
    """Flow matching ODE solver.

    The drift for the ODE is the velocity field itself:

    .. math::
        dx = u_t(x)\\, dt

    Parameters
    ----------
    velocity_model : ModelWrapper
        Wrapped velocity field model.

    Example
    -------
    .. code-block:: python

        from gensbi.flow_matching.solver.fm_ode_solver import FMODESolver
        from gensbi.utils.model_wrapping import ModelWrapper
        import jax.numpy as jnp

        model_wrapped = ModelWrapper(my_velocity_model)
        solver = FMODESolver(velocity_model=model_wrapped)
        sol = solver.sample(x_init, step_size=0.01, time_grid=jnp.array([0.0, 1.0]))
    """

    def get_drift(self, **kwargs) -> Callable:
        """Return the velocity field as the ODE drift."""
        return self.velocity_model.get_vector_field(**kwargs)
