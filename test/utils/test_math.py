import os
os.environ['JAX_PLATFORMS']="cpu"

import jax.numpy as jnp
import jax
import pytest

from gensbi.utils.math import divergence, _expand_dims, _expand_time

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


def test_expand_dims():
    """Test internal dimension expansion for rank 1, 2, and 3 inputs."""
    # 1D array: (N,) -> (1, N, 1)
    x1 = jnp.arange(5)
    res1 = _expand_dims(x1)
    assert res1.ndim == 3
    assert res1.shape == (1, 5, 1)
    assert jnp.array_equal(res1.flatten(), x1)

    # 2D array: (N, D) -> (N, D, 1)
    x2 = jnp.arange(10).reshape(5, 2)
    res2 = _expand_dims(x2)
    assert res2.ndim == 3
    assert res2.shape == (5, 2, 1)
    assert jnp.array_equal(res2.flatten(), x2.flatten())

    # 3D array: (N, D, C) -> (N, D, C) (unchanged)
    x3 = jnp.arange(30).reshape(5, 2, 3)
    res3 = _expand_dims(x3)
    assert res3.ndim == 3
    assert res3.shape == (5, 2, 3)
    assert jnp.array_equal(res3, x3)


def test_expand_time():
    """Test internal time expansion for rank 0, 1, and 2 inputs."""
    # Scalar: () -> (1, 1)
    t0 = jnp.array(0.5)
    res0 = _expand_time(t0)
    assert res0.ndim == 2
    assert res0.shape == (1, 1)
    assert res0[0, 0] == t0

    # 1D array: (N,) -> (N, 1)
    t1 = jnp.array([0.1, 0.2])
    res1 = _expand_time(t1)
    assert res1.ndim == 2
    assert res1.shape == (2, 1)
    assert jnp.array_equal(res1.flatten(), t1)

    # 2D array: (N, 1) -> (N, 1) (unchanged)
    t2 = jnp.array([[0.1], [0.2]])
    res2 = _expand_time(t2)
    assert res2.ndim == 2
    assert res2.shape == (2, 1)
    assert jnp.array_equal(res2, t2)
