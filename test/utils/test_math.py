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


@pytest.mark.parametrize(
    "input_array, expected_shape",
    [
        (jnp.arange(5), (1, 5, 1)),  # 1D array: (N,) -> (1, N, 1)
        (jnp.arange(10).reshape(5, 2), (5, 2, 1)),  # 2D array: (N, D) -> (N, D, 1)
        (jnp.arange(30).reshape(5, 2, 3), (5, 2, 3)),  # 3D array: (N, D, C) -> (N, D, C)
    ],
    ids=["1D", "2D", "3D"],
)
def test_expand_dims(input_array, expected_shape):
    """Test internal dimension expansion for various input ranks."""
    res = _expand_dims(input_array)
    assert res.ndim == 3
    assert res.shape == expected_shape
    if input_array.ndim < 3:
        assert jnp.array_equal(res.flatten(), input_array.flatten())
    else:
        assert jnp.array_equal(res, input_array)


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
