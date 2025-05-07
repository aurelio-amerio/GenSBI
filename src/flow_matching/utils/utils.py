#FIXME: first pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import Array


def unsqueeze_to_match(source: Array, target: Array, how: str = "suffix") -> Array:
    """
    Unsqueeze the source array to match the dimensionality of the target array.

    Args:
        source (Array): The source array to be unsqueezed.
        target (Array): The target array to match the dimensionality of.
        how (str, optional): Whether to unsqueeze the source array at the beginning
            ("prefix") or end ("suffix"). Defaults to "suffix".

    Returns:
        Array: The unsqueezed source array.
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

    Args:
        input_array (Array): (batch_size,).
        expand_to (Array): (batch_size, ...).

    Returns:
        Array: (batch_size, ...).
    """
    assert len(input_array.shape) == 1, "Input array must be a 1d vector."
    assert (
        input_array.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_array.shape} and {expand_to.shape}."

    dim_diff = len(expand_to.shape) - len(input_array.shape)
    
    t_expanded = jnp.reshape(input_array, (-1,) + (1,) * dim_diff)
    return jnp.broadcast_to(t_expanded, expand_to.shape)


# def gradient(
#     output: Array,
#     x: Array,
#     grad_outputs: Optional[Array] = None,
# ) -> Array:
#     """
#     Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

#     Args:
#         output (Array): [N, D] Output of the function.
#         x (Array): [N, d_1, d_2, ... ] input
#         grad_outputs (Optional[Array]): [N, D] Gradient of outputs, if `None`,
#             then will use an array of ones
#         create_graph (bool): If True, graph of the derivative will be constructed, allowing
#             to compute higher order derivative products. Defaults to False.
#     Returns:
#         Array: [N, d_1, d_2, ... ]. the gradient w.r.t x.
#     """
#     if grad_outputs is None:
#         grad_outputs = jnp.ones_like(output)
    
#     # Use JAX's grad with vjp for custom gradients
#     def inner_product(x):
#         return jnp.sum(output * grad_outputs)
    
#     return jax.grad(inner_product)(x)

def _divergence_single(vf, x, t):
    res = jnp.trace(jax.jacfwd(vf, argnums=0)(x, t))
    return res

    


def divergence(
        vf: Callable, 
        x: Array,
        t: Array,
        ):
    """
    Compute the divergence of the vector field vf at point x and time t.
    Args:
        vf (Callable): The vector field function.
        x (Array): The point at which to compute the divergence.
        t (Array): The time at which to compute the divergence.
    Returns:
        Array: The divergence of the vector field at point x and time t.
    """
    x = jnp.atleast_2d(x)
    t = jnp.atleast_1d(t)
    if len(t.shape) < 2:
        t = t[..., None]
        t = jnp.broadcast_to(t, (x.shape[0], t.shape[-1]))

    return jax.vmap(_divergence_single, in_axes=(None, 0, 0))(vf, x, t)