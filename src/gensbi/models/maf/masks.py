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

    Parameters
    ----------
    in_ranks : Array
        Integer rank assigned to each input unit; shape ``(in_features,)``.
        Units with rank -1 (conditioning inputs) are allowed to feed every
        output unit.
    out_ranks : Array
        Integer rank assigned to each output unit; shape ``(out_features,)``.
    strict : bool
        If ``True``, use strict inequality (``out_rank > in_rank``), which is
        required for the output layer to enforce the autoregressive property.
        If ``False``, use non-strict inequality (``out_rank >= in_rank``),
        which is used for hidden layers so that units of the same rank may
        communicate.

    Returns
    -------
    Array
        Boolean mask of shape ``(in_features, out_features)`` where entry
        ``[i, o]`` is ``True`` iff input unit ``i`` is permitted to influence
        output unit ``o`` under the chosen rank inequality.
    """
    op = operator.gt if strict else operator.ge
    return op(out_ranks[None, :], in_ranks[:, None])
