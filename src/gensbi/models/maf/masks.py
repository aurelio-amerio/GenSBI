"""Rank-based binary masks for masked autoregressive networks."""

import operator

import jax.numpy as jnp
from jax import Array


def make_mask(in_ranks: Array, out_ranks: Array, *, strict: bool) -> Array:
    """Binary connectivity mask of shape ``(len(in_ranks), len(out_ranks))``.

    ``mask[i, o]`` is True iff input unit ``i`` may feed output unit ``o``:
    ``out_ranks[o] > in_ranks[i]`` when ``strict`` (final/output layer), else
    ``out_ranks[o] >= in_ranks[i]`` (hidden layers). The ``(in, out)`` layout
    matches an ``nnx.Linear`` kernel so it multiplies the weight directly.
    """
    op = operator.gt if strict else operator.ge
    return op(out_ranks[None, :], in_ranks[:, None])
