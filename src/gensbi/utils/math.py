import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, Optional
from einops import rearrange, einsum

def _expand_dims(x: Array) -> Array:
    if x.ndim < 3:
        x = rearrange(x, "... -> 1 ... 1" if x.ndim == 1 else "... -> ... 1")
    return x


def _expand_time(t: Array) -> Array:
    t = jnp.atleast_1d(t)
    if t.ndim < 2:
        t = t[..., None]
    return t


# def _divergence_single(vf, t, x):
#     res = jax.jacfwd(vf, argnums=1)(t, x)
#     res = rearrange(res, ' b c d e ->  (b c) (d e)')
#     res = jnp.trace(res, axis1=-2, axis2=-1)
#     return res


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
    x = _expand_dims(x)
    t = _expand_time(t)

    vf_wrapped = lambda t, x: vf(t, x, args=args)

    # res = jax.vmap(_divergence_single, in_axes=(None, 0, 0))(vf_wrapped, t, x)

    # res = jax.vmap(jax.jacfwd(vf_wrapped, argnums=1), in_axes=(0,0))(t, x)
    # res = rearrange(res, 'a b c d e -> a (b c) (d e)')
    # res = jnp.trace(res, axis1=-2, axis2=-1)
    # return jnp.squeeze(res, axis=1) if res.ndim > 1 else res

    res = jax.jacfwd(vf_wrapped, argnums=1)(t, x)
    res = rearrange(res, 'i a b j c d -> i (a b) j (c d)')
    res = einsum(res, 'i a i c -> i')
    return jnp.squeeze(res)
