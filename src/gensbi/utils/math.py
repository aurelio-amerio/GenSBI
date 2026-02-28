"""
Mathematical utility functions for GenSBI.

This module provides mathematical operations and transformations used throughout
the library, including dimension expansion and divergence computation for vector fields.
"""
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, Optional
from einops import rearrange


def _expand_dims(x: Array) -> Array:
    """
    Expand dimensions of an array to have at least 3 dimensions.
    
    Parameters
    ----------
        x: Input array to expand.
        
    Returns
    -------
        Array with at least 3 dimensions.
    """
    if x.ndim < 3:
        x = rearrange(x, "... -> 1 ... 1" if x.ndim == 1 else "... -> ... 1")
    return x


def _expand_time(t: Array) -> Array:
    """
    Expand time array to have at least 2 dimensions.
    
    Parameters
    ----------
        t: Time array to expand.
        
    Returns
    -------
        Time array with at least 2 dimensions.
    """
    t = jnp.atleast_1d(t)
    if t.ndim < 2:
        t = t[..., None]
    return t



def divergence(
    vf: Callable,
    t: Array,
    x: Array,
    args: Optional[Array] = None,
) -> Array:
    """
    Compute the divergence of a vector field at specified points and times.

    Uses ``vmap(jacfwd)`` to compute per-sample Jacobians efficiently,
    then takes the trace over the flattened ``(features, channels)`` axes.

    Parameters
    ----------
        vf: The vector field function with signature ``vf(t, x, args)``.
        t: The time at which to compute the divergence.
        x: The point at which to compute the divergence.
        args: Optional additional arguments for the vector field function.

    Returns
    -------
        The divergence of the vector field at point x and time t,
        with shape ``(batch,)``.
    """
    x = _expand_dims(x)
    t = _expand_time(t)

    # Broadcast t to match x's batch dimension for vmap
    batch_size = x.shape[0]
    t = jnp.broadcast_to(t, (batch_size, t.shape[-1]))

    def _single_div(t_i, x_i):
        """Divergence for a single sample (no batch dimension)."""
        # x_i: (features, channels), t_i: (1,)
        def f(x_):
            # Unsqueeze batch dim, call vf, squeeze back
            return vf(t_i[None], x_[None], args=args)[0]
        jac = jax.jacfwd(f)(x_i)  # (features, channels, features, channels)
        jac = rearrange(jac, 'a b c d -> (a b) (c d)')  # (D*C, D*C)
        return jnp.trace(jac)

    return jax.vmap(_single_div)(t, x)


# NOTE: When using divergence_hutchinson inside an ODE solve for log-probability
# computation, the probe vector v should be drawn ONCE before the ODE solve and
# reused at every time step. A fixed probe gives smoother augmented dynamics,
# which is critical for adaptive solvers (Dopri5) and reduces variance in the
# accumulated log-det integral.
# Reference: Meta's flow_matching library `compute_likelihood` draws z once:
#   z = (torch.randn_like(x_1) < 0) * 2.0 - 1.0  # fixed before odeint
# The caller should generate the probe and pass it through the ODE args.
def divergence_hutchinson(
    vf: Callable,
    t: Array,
    x: Array,
    args: Optional[Array] = None,
    *,
    v: Array = None,
) -> Array:
    """
    Estimate the divergence of a vector field using the Hutchinson trace estimator.

    Uses a single JVP with a probe vector to obtain an unbiased estimate of
    tr(J), where J = ∂vf/∂x:

        tr(J) ≈ vᵀ J v

    The probe vector ``v`` should be drawn externally (e.g., Rademacher ±1)
    and fixed across ODE steps for lower-variance log-probability estimates.

    Parameters
    ----------
        vf: The vector field function with signature ``vf(t, x, args)``.
        t: The time at which to compute the divergence.
        x: The point at which to compute the divergence.
        args: Optional additional arguments for the vector field function.
        v: Probe vector, same shape as x (after ``_expand_dims``).
            Typically Rademacher ±1, drawn once and reused across ODE steps.

    Returns
    -------
        The Hutchinson estimate of the divergence at point x and time t.
    """
    x = _expand_dims(x)
    t = _expand_time(t)

    vf_wrapped = lambda x_: vf(t, x_, args=args)

    # JVP: (vf(x), J @ v) in a single forward pass
    _, jvp_val = jax.jvp(vf_wrapped, (x,), (v,))

    # Hutchinson estimate: vᵀ (J v) summed per sample
    estimate = jnp.sum(v * jvp_val, axis=tuple(range(1, v.ndim)))

    return jnp.squeeze(estimate)
