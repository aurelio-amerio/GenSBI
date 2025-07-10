import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.path_sample import EDMPathSample
from jax import numpy as jnp


def test_path_sample_initialization():
    # Minimal test, as EDMPathSample likely requires more context
    sample = EDMPathSample(jnp.zeros((1, 1)),jnp.zeros((1, 1)),jnp.zeros((1, 1)))  # Example shape
    assert isinstance(sample, EDMPathSample)
