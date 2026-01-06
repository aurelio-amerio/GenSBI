"""
Utility functions for flow matching.

This module provides helper functions for tensor manipulation and array operations
commonly used in flow matching algorithms, including dimension expansion and broadcasting.
"""
from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import Array

import matplotlib.pyplot as plt
import numpy as np

from einops import einsum


def unsqueeze_to_match(source: Array, target: Array, how: str = "suffix") -> Array:
    """
    Unsqueeze the source array to match the dimensionality of the target array.

    Parameters
    ----------
        source : Array
            The source array to be unsqueezed.
        target : Array
            The target array to match the dimensionality of.
        how : str, optional
            Whether to unsqueeze the source array at the beginning
            ("prefix") or end ("suffix"). Defaults to "suffix".

    Returns
    -------
        Array
            The unsqueezed source array.
    """
    assert (
        how == "prefix" or how == "suffix"
    ), f"{how} is not supported, only 'prefix' and 'suffix' are supported."

    dim_diff = len(target.shape) - len(source.shape)

    for _ in range(dim_diff):
        if how == "prefix":
            source = jnp.expand_dims(source, axis=0)
        elif how == "suffix":
            source = jnp.expand_dims(source, axis=-1)

    return source


def expand_tensor_like(input_array: Array, expand_to: Array) -> Array:
    """`input_array` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_array` to have the same shape as `expand_to` along all remaining dimensions.

    Parameters
    ----------
        input_array : Array
            (batch_size,).
        expand_to : Array
            (batch_size, ...).

    Returns
    -------
        Array
            (batch_size, ...).
    """
    assert len(input_array.shape) == 1, "Input array must be a 1d vector."
    assert (
        input_array.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_array.shape} and {expand_to.shape}."

    dim_diff = len(expand_to.shape) - len(input_array.shape)

    t_expanded = jnp.reshape(input_array, (-1,) + (1,) * dim_diff)
    return jnp.broadcast_to(t_expanded, expand_to.shape)
