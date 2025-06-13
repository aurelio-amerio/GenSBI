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

import matplotlib.pyplot as plt
import numpy as np

from einops import einsum


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

def _divergence_single(vf, t, x):
    res = jnp.trace(jax.jacfwd(vf, argnums=1)(t, x),axis1=-2, axis2=-1)
    return res

    
def divergence(
        vf: Callable, 
        t: Array,
        x: Array,
        args: Optional[Array] = None,
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
    x = jnp.atleast_1d(x)
    if x.ndim < 2: 
        x = jnp.expand_dims(x, axis=0)
    t = jnp.atleast_1d(t)
    t = jnp.broadcast_to(
        t, (*x.shape[:-1], t.shape[-1])
    )

    # vf_wapped = lambda t, x: vf(t, x, args=args)

    # res = jax.vmap(_divergence_single, in_axes=(None, 0, 0))(vf_wapped, t, x)

    vf_wrapped = lambda t, x: vf(t, x, args=args)

    res = jax.vmap(_divergence_single, in_axes=(None, 0, 0))(vf_wrapped, t, x)

    return jnp.squeeze(res)
    
# def divergence(
#         vf: Callable, 
#         t: Array,
#         x: Array,
#         args: Optional[Array] = None,
#         ):
#     """
#     Compute the divergence of the vector field vf at point x and time t.
#     Args:
#         vf (Callable): The vector field function.
#         x (Array): The point at which to compute the divergence.
#         t (Array): The time at which to compute the divergence.
#     Returns:
#         Array: The divergence of the vector field at point x and time t.
#     """
#     x = jnp.atleast_2d(x)
#     t = jnp.atleast_1d(t)
#     if len(t.shape) < 2:
#         for i in range(len(x.shape) - 1):
#             t = jnp.expand_dims(t, axis=-1)
#         t = jnp.broadcast_to(t, (*x.shape[:-1], t.shape[-1]))

#     trace = jax.jacfwd(vf, argnums=1)(t, x, args=args)
#     res = einsum(trace, "b i i ... -> b ...")
#     return res


# plotting utils
def plot_trajectories(traj):
    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(traj[0,:,0], traj[0,:,1], color="red", s=1, alpha=1)
    ax.plot(traj[:,:,0], traj[:,:,1], color="white", lw=0.5, alpha=0.7)
    ax.scatter(traj[-1,:,0], traj[-1,:,1], color="blue", s=2, alpha=1, zorder=2)
    ax.set_aspect('equal', adjustable='box')
    # set black background
    ax.set_facecolor('#A6AEBF')
    return fig, ax