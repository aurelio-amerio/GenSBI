"""Shared dtype-assertion helpers for mixed-precision tests."""

import jax.numpy as jnp
import flax.traverse_util as tu


def float_leaves(tree):
    """Return {'.'-joined path: dtype} for every floating-point leaf."""
    flat = tu.flatten_dict(tree)
    return {
        ".".join(str(p) for p in path): leaf.dtype
        for path, leaf in flat.items()
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    }


def assert_tree_dtype(tree, dtype):
    """Assert every floating leaf of ``tree`` has exactly ``dtype``."""
    offenders = {k: d for k, d in float_leaves(tree).items() if d != jnp.dtype(dtype)}
    assert not offenders, (
        f"expected all floating leaves to be {jnp.dtype(dtype)}, got: {offenders}"
    )
