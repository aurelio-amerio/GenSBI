import os
os.environ['JAX_PLATFORMS']="cpu"

import jax.numpy as jnp
import jax
import pytest

from gensbi.utils.math import divergence

def test_divergence_linear_field():
    # vf(t, x, args) = A @ x, where A is a constant matrix
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    def vf(t, x, args=None):
        return A * x

    t = jnp.array([0.5])
    x = jnp.array([1.0, 2.0]).reshape(1,2,1)
    div = divergence(vf, t, x)
    # For a linear field, divergence is the trace of A
    assert jnp.allclose(div, 5.0), f"Expected divergence 5.0, got {div}"

# def test_divergence_single():
#     # vf(t, x, args) = x, divergence should be 2 for 2D
#     def vf(t, x, args=None):
#         return x

#     t = jnp.array([0.1])
#     x = jnp.array([1.0, 2.0])
#     div = divergence(vf, t, x)
#     assert div.shape == (1,)

def test_divergence_batch():
    # vf(t, x, args) = x, divergence should be 2 for 2D
    def vf(t, x, args=None):
        return x

    t = jnp.array([0.1, 0.2])
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    div = divergence(vf, t, x)
    assert div.shape == (2,)
    assert jnp.allclose(div, 2.0)

def test_divergence_with_args():
    # vf(t, x, args) = args * x, args is a scalar
    def vf(t, x, args=None):
        return args * x

    t = jnp.array([0.0])
    x = jnp.array([[1.0, 2.0]])
    args = 4.0
    div = divergence(vf, t, x, args=args)
    # divergence should be 4 + 4 = 8
    assert jnp.allclose(div, 8.0)
