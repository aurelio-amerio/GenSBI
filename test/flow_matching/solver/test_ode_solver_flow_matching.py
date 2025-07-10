import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest
from gensbi.flow_matching.solver.ode_solver import ODESolver
from gensbi.utils.model_wrapping import ModelWrapper

from flax import nnx

class DummyModel(nnx.Module):
    def __call__(self, x, t, args, **kwargs):
        return jnp.ones_like(x) * 3.0 * t**2


@pytest.fixture
def solver():
    dummy_model = DummyModel()
    dummy_wrapped_model = ModelWrapper(dummy_model)
    return ODESolver(velocity_model=dummy_wrapped_model)

def test_sample_shape(solver):
    x_init = jnp.ones((5, 2))
    time_grid = jnp.array([0.0, 1.0])
    sol = solver.sample(time_grid=time_grid, x_init=x_init, method='Dopri5', step_size=0.1, return_intermediates=False)
    assert sol.shape == x_init.shape

    time_grid = jnp.linspace(0,1,10)
    sol = solver.sample(time_grid=time_grid, x_init=x_init, method='Dopri5', step_size=0.1, return_intermediates=True)
    assert sol.shape == (10, *x_init.shape)
