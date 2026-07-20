import jax.numpy as jnp
import pytest
from flax import nnx

from tests.precision_utils import assert_tree_dtype, float_leaves


def test_assert_tree_dtype_passes_on_uniform_tree():
    tree = {"a": jnp.ones((2,), jnp.float32), "b": {"c": jnp.zeros((3,), jnp.float32)}}
    assert_tree_dtype(tree, jnp.float32)


def test_assert_tree_dtype_ignores_non_float_leaves():
    tree = {"ids": jnp.zeros((2,), jnp.int32), "w": jnp.ones((2,), jnp.float32)}
    assert_tree_dtype(tree, jnp.float32)


def test_assert_tree_dtype_fails_and_names_offender():
    tree = {"good": jnp.ones((2,), jnp.float32), "bad": jnp.ones((2,), jnp.bfloat16)}
    with pytest.raises(AssertionError, match="bad"):
        assert_tree_dtype(tree, jnp.float32)


def test_float_leaves_reports_dtypes():
    tree = {"w": jnp.ones((2,), jnp.bfloat16)}
    assert float_leaves(tree) == {"w": jnp.bfloat16}


def test_works_on_nnx_param_state():
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    tree = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    assert_tree_dtype(tree, jnp.float32)
