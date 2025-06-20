
import jax 
import jax.numpy as jnp
from jax import Array
from typing import Callable, Optional

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

    vf_wrapped = lambda t, x: vf(t, x, args=args)

    res = jax.vmap(_divergence_single, in_axes=(None, 0, 0))(vf_wrapped, t, x)

    return jnp.squeeze(res)
    