import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.path_sample import EDMPathSample
from jax import numpy as jnp


def test_path_sample_initialization():
    # Minimal test, as EDMPathSample likely requires more context
    sample = EDMPathSample(jnp.zeros((1, 1)),jnp.zeros((1, 1)),jnp.zeros((1, 1)))  # Example shape
    assert isinstance(sample, EDMPathSample)


def test_path_sample_get_batch():
    sample = EDMPathSample(jnp.ones((2, 2)), jnp.zeros((2, 1)), jnp.full((2, 2), 5.0))
    x_1, x_t, sigma = sample.get_batch()
    assert (x_1 == jnp.ones((2, 2))).all()
    assert (x_t == jnp.full((2, 2), 5.0)).all()
    assert (sigma == jnp.zeros((2, 1))).all()
