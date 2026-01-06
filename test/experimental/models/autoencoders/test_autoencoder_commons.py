import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import jax
from flax import nnx
import pytest

from gensbi.experimental.models.autoencoders.commons import DiagonalGaussian

def test_DiagonalGaussian():
    batch_size = 12
    z_channels = 4
    latent = jnp.zeros((batch_size, z_channels))

    diag_gauss = DiagonalGaussian(sample=True)
    key = jax.random.PRNGKey(0)
    res = diag_gauss(latent, key)

    assert res.shape == (batch_size, z_channels//2)
    
    
    diag_gauss = DiagonalGaussian(sample=False)
    res = diag_gauss(latent)
    
    assert res.shape == (batch_size, z_channels//2)
    return