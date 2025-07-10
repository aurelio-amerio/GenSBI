import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest
from gensbi.flow_matching.path.scheduler.scheduler import CondOTScheduler, SchedulerOutput

@pytest.fixture
def scheduler():
    return CondOTScheduler()

def test_scheduler_output_shapes(scheduler):
    t = jnp.array([0.1, 0.5, 0.9])
    output = scheduler(t)
    expected_shape = t.shape
    assert output.alpha_t.shape == expected_shape
    assert output.sigma_t.shape == expected_shape
    assert output.d_alpha_t.shape == expected_shape
    assert output.d_sigma_t.shape == expected_shape

def test_kappa_inverse(scheduler):
    t = jnp.array([0.1, 0.5, 0.9])
    output = scheduler(t)
    t_recovered = scheduler.kappa_inverse(output.alpha_t)
    assert jnp.allclose(t, t_recovered, atol=1e-5)
