import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.flow_matching.path.affine import AffineProbPath
from gensbi.flow_matching.path.scheduler.scheduler import CondOTScheduler

@pytest.fixture
def affine_prob_path():
    scheduler = CondOTScheduler()
    return AffineProbPath(scheduler)

def test_affine_prob_path_sample(affine_prob_path):
    batch_size, dim = 10, 5
    x_0 = jnp.ones((batch_size, dim))
    x_1 = jnp.ones((batch_size, dim)) * 2
    t = jnp.ones((batch_size,)) * 0.5
    sample = affine_prob_path.sample(x_0, x_1, t)
    # Check all returned shapes
    assert sample.x_t.shape == (batch_size, dim)
    assert sample.dx_t.shape == (batch_size, dim)
    assert sample.x_0.shape == (batch_size, dim)
    assert sample.x_1.shape == (batch_size, dim)
    assert sample.t.shape == (batch_size,)
    # Check values
    assert jnp.all(sample.t == t)
    assert jnp.all(sample.x_0 == x_0)
    assert jnp.all(sample.x_1 == x_1)

def test_assert_sample_shape(affine_prob_path):
    batch_size, dim = 10, 5
    x_0 = jnp.ones((batch_size, dim))
    x_1 = jnp.ones((batch_size, dim))
    t = jnp.ones((batch_size,))
    # Should not raise
    affine_prob_path.assert_sample_shape(x_0, x_1, t)
