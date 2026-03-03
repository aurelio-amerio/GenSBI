"""
Score matching SDE samplers.

This module implements reverse SDE sampling for standard score matching models
using diffrax for numerical integration.

Based on "Score-Based Generative Modeling through Stochastic Differential Equations"
by Song et al., 2021. https://arxiv.org/abs/2011.13456
"""

from typing import Any, Callable, Optional
import math

import jax
import jax.numpy as jnp
from jax import Array, jit

import diffrax
from diffrax import (
    AbstractERK,
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
    SaveAt,
)


def sm_reverse_sde_sampler(
    sde: Any,
    score_model: Callable,
    x_init: Array,
    *,
    key: Array,
    condition_mask: Optional[Array] = None,
    condition_value: Optional[Array] = None,
    return_intermediates: bool = False,
    n_steps: int = 1000,
    eps: float = 1e-3,
    method: str = "Euler",
    model_kwargs: dict = {},
) -> Array:
    r"""
    Sample from the reverse SDE using diffrax integration.

    **Time direction convention:**
    The forward SDE runs from ``t=0`` (clean data) to ``t=T`` (noise).
    This reverse sampler integrates **backwards** from ``t=T`` to ``t=eps``
    to generate samples. This is the opposite direction from flow matching,
    which integrates ``t=0`` to ``t=1`` to go from noise to data.

    **Input shape convention:**
    All inputs must have shape ``(batch, features, channels)``.
    If your data is 2D ``(batch, features)``, add a trailing dimension:
    ``x = x[..., None]``.

    The reverse SDE is:

    .. math::
        dx = \left[ f(x,t) - g(t)^2 s_\theta(x, t) \right] dt + g(t) d\bar{W}

    where :math:`s_\theta` is the learned score model, :math:`f` is the forward drift,
    and :math:`g` is the forward diffusion coefficient.

    Parameters
    ----------
        sde : Any
            The forward SDE scheduler (VPSmScheduler or VESmScheduler).
        score_model : Callable
            The score model function, called as ``score_model(obs=x, t=t, **model_kwargs)``.
        x_init : Array
            Initial samples from the prior, shape ``(batch_size, features, channels)``.
        key : Array
            JAX random key.
        condition_mask : Optional[Array]
            Boolean mask for conditioning (True for conditioned dimensions).
        condition_value : Optional[Array]
            Values for conditioned dimensions.
        return_intermediates : bool
            Whether to return all intermediate steps.
        n_steps : int
            Number of integration steps.
        eps : float
            Minimum time (to avoid singularities near t=0).
        method : str
            Integration method. One of ``"Euler"``, ``Heun``, ``"SEA"``, ``"ShARK"``.
        model_kwargs : dict
            Additional keyword arguments passed to the score model.

    Returns
    -------
        Array
            Sampled output. Shape ``(batch_size, features, channels)`` if
            ``return_intermediates`` is False, or
            ``(n_steps+1, batch_size, features, channels)`` if True.
    """
    assert x_init.ndim == 3, (
        f"x_init must have shape (batch, features, channels), got {x_init.shape}. "
        "If your data is 2D, use x_init[..., None]."
    )
    assert (
        condition_mask is None or condition_value is not None
    ), "Condition value must be provided if condition mask is provided"

    solvers = {
        "Euler": diffrax.Euler,
        "Heun": diffrax.Heun,
        "SEA": diffrax.SEA,
        "ShARK": diffrax.ShARK,
    }
    assert (
        method in solvers
    ), f"Unknown method '{method}'. Choose from {list(solvers.keys())}."

    if method in ["Euler", "Heun"]:
        levy_area = diffrax.BrownianIncrement
    else:
        levy_area = diffrax.SpaceTimeLevyArea

    solver = solvers[method]()

    # Adaptive step sizing: ShARK automatically uses PIDController
    if method == "ShARK":
        dtmin = min(2e-5, abs(-(sde.T - eps) / n_steps))
        stepsize_controller = diffrax.PIDController(
            rtol=1e-5, atol=1e-5, dtmin=dtmin, dtmax=2 * abs(-(sde.T - eps) / n_steps)
        )
    else:
        stepsize_controller = diffrax.ConstantStepSize()

    # Time direction: reverse SDE goes from t=T (noise) to t=eps (near-clean data).
    # This is OPPOSITE to flow matching, which goes t=0 → t=1.
    t0 = sde.T  # start at noise time
    t1 = eps  # stop near clean data
    dt0 = -(t0 - t1) / n_steps  # negative dt: integrating backwards in time

    batch_size = x_init.shape[0]
    sample_shape = x_init.shape[1:]  # e.g. (dim,) or (features, channel)
    flat_dim = math.prod(sample_shape)

    # Apply conditioning to initial samples
    if condition_mask is not None:
        x_init = x_init * (1 - condition_mask) + condition_value * condition_mask
        # Flatten mask for per-sample use
        condition_mask_flat = condition_mask.reshape(batch_size, flat_dim)
    else:
        condition_mask_flat = None

    def make_reverse_drift(cond_mask_i):
        """Create reverse drift closure for a single sample."""

        def reverse_drift(t, y_flat, args):
            # y_flat: (flat_dim,) -> reshape to sample_shape for model
            y = y_flat.reshape(sample_shape)
            # Add batch dim for model call; pass t as scalar (model handles expansion)
            y_batched = y[None, ...]  # (1, ...)
            t_batched = jnp.atleast_1d(t)[None, ...]  # (1, 1)
            score = score_model(obs=y_batched, t=t_batched, **args)
            score = jnp.squeeze(score, axis=0)  # remove batch dim

            # Broadcast t to spatial shape only for SDE coefficient computations
            t_broadcast = jnp.broadcast_to(t, y.shape)
            g_sq = sde.diffusion(t_broadcast) ** 2
            forward_drift = sde.drift(y, t_broadcast)
            result = forward_drift - g_sq * score.reshape(sample_shape)
            if cond_mask_i is not None:
                result = result * (1 - cond_mask_i.reshape(sample_shape))
            return result.reshape(flat_dim)

        return reverse_drift

    def make_reverse_diffusion(cond_mask_i):
        """Create reverse diffusion closure for a single sample.
        Returns a (flat_dim, flat_dim) matrix for ControlTerm.
        """

        def reverse_diffusion(t, y_flat, args):
            t_broadcast = jnp.broadcast_to(t, (flat_dim,))
            g = sde.diffusion(t_broadcast)
            if cond_mask_i is not None:
                g = g * (1 - cond_mask_i.reshape(flat_dim))
            return jnp.diag(g)

        return reverse_diffusion

    def sample_one(key_i, y0_flat, cond_mask_i):
        tol = min(2e-5, abs(dt0))

        brownian_motion = VirtualBrownianTree(
            t1,
            t0,
            tol=tol,
            shape=(flat_dim,),
            key=key_i,
            levy_area=levy_area,
        )
        drift_fn = make_reverse_drift(cond_mask_i)
        diff_fn = make_reverse_diffusion(cond_mask_i)

        terms = MultiTerm(
            ODETerm(drift_fn),
            ControlTerm(diff_fn, brownian_motion),
        )

        if return_intermediates:
            saveat = SaveAt(ts=jnp.linspace(t0, t1, n_steps + 1))
        else:
            saveat = SaveAt(t1=True)

        sol = diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0=dt0,
            y0=y0_flat,
            args=model_kwargs,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        return sol.ys  # (n_saves, flat_dim)

    # Flatten inputs for diffrax
    x_init_flat = x_init.reshape(batch_size, flat_dim)

    keys = jax.random.split(key, batch_size)

    if condition_mask_flat is not None:
        results = jax.vmap(sample_one)(keys, x_init_flat, condition_mask_flat)
    else:
        # Use None broadcast via vmap in_axes
        results = jax.vmap(sample_one, in_axes=(0, 0, None))(keys, x_init_flat, None)

    # results shape: (batch, n_saves, flat_dim)
    if return_intermediates:
        # (batch, n_steps+1, flat_dim) -> (n_steps+1, batch, *sample_shape)
        results = results.reshape(batch_size, n_steps + 1, *sample_shape)
        ndim = len(sample_shape)
        perm = (1, 0) + tuple(range(2, 2 + ndim))
        return jnp.transpose(results, perm)
    else:
        # (batch, 1, flat_dim) -> (batch, *sample_shape)
        return results.reshape(batch_size, *sample_shape)


def sm_reverse_ode_sampler(
    sde: Any,
    score_model: Callable,
    x_init: Array,
    *,
    key: Array,
    condition_mask: Optional[Array] = None,
    condition_value: Optional[Array] = None,
    return_intermediates: bool = False,
    n_steps: int = 1000,
    eps: float = 1e-3,
    method: str = "Euler",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    model_kwargs: dict = {},
) -> Array:
    r"""
    Sample from the probability flow ODE (PF-ODE) using diffrax integration.

    Implements the deterministic sampler based on the Probability Flow ODE
    from section 4.3 of "Score-Based Generative Modeling through Stochastic
    Differential Equations" by Song et al., 2021.
    See https://arxiv.org/abs/2011.13456.

    Unlike the reverse SDE sampler, this ODE has no stochastic (diffusion)
    term, making it fully deterministic for a given ``x_init``. This
    enables exact log_prob computation.

    **Time direction convention:**
    The forward SDE runs from ``t=0`` (clean data) to ``t=T`` (noise).
    This sampler integrates **backwards** from ``t=T`` to ``t=eps``
    to generate samples.

    **Input shape convention:**
    All inputs must have shape ``(batch, features, channels)``.
    If your data is 2D ``(batch, features)``, add a trailing dimension:
    ``x = x[..., None]``.

    The PF-ODE is:

    .. math::
        dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x, t) \right] dt

    where :math:`s_\theta` is the learned score model, :math:`f` is the
    forward drift, and :math:`g` is the forward diffusion coefficient.
    Compared to the reverse SDE, the diffusion coefficient is halved and
    the Brownian noise term is removed.

    Parameters
    ----------
        sde : Any
            The forward SDE scheduler (VPSmScheduler or VESmScheduler).
        score_model : Callable
            The score model function, called as ``score_model(obs=x, t=t, **model_kwargs)``.
        x_init : Array
            Initial samples from the prior, shape ``(batch_size, features, channels)``.
        key : Array
            JAX random key (unused by the ODE itself, kept for API consistency).
        condition_mask : Optional[Array]
            Boolean mask for conditioning (True for conditioned dimensions).
        condition_value : Optional[Array]
            Values for conditioned dimensions.
        return_intermediates : bool
            Whether to return all intermediate steps.
        n_steps : int
            Number of integration steps.
        eps : float
            Minimum time (to avoid singularities near t=0).
        method : str
            Integration method. One of ``"Euler"``, ``"Heun"``, ``"Dopri5"``.
            ``"Dopri5"`` automatically uses adaptive step sizing via
            ``PIDController``; the others use fixed step size.
        atol : float
            Absolute tolerance for adaptive step solvers.
        rtol : float
            Relative tolerance for adaptive step solvers.
        model_kwargs : dict
            Additional keyword arguments passed to the score model.

    Returns
    -------
        Array
            Sampled output. Shape ``(batch_size, features, channels)`` if
            ``return_intermediates`` is False, or
            ``(n_steps+1, batch_size, features, channels)`` if True.
    """

    assert x_init.ndim == 3, (
        f"x_init must have shape (batch, features, channels), got {x_init.shape}. "
        "If your data is 2D, use x_init[..., None]."
    )
    assert (
        condition_mask is None or condition_value is not None
    ), "Condition value must be provided if condition mask is provided"

    solvers = {
        "Euler": diffrax.Euler,
        "Heun": diffrax.Heun,
        "Dopri5": diffrax.Dopri5,
    }
    assert (
        method in solvers
    ), f"Unknown method '{method}'. Choose from {list(solvers.keys())}."

    solver = solvers[method]()

    # Adaptive step sizing: only Dopri5 uses PIDController (high-order embedded
    # error pair makes adaptivity worthwhile). Heun stays fixed-step because its
    # low-order error estimator doesn't justify the overhead given expensive
    # score model evaluations per step.
    if isinstance(solver, diffrax.Dopri5):
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    else:
        stepsize_controller = diffrax.ConstantStepSize()

    # Time direction: reverse ODE goes from t=T (noise) to t=eps (near-clean data).
    # This is OPPOSITE to flow matching, which goes t=0 → t=1.
    t0 = sde.T  # start at noise time
    t1 = eps  # stop near clean data
    dt0 = -(t0 - t1) / n_steps  # negative dt: integrating backwards in time
    # For adaptive solvers, let the controller pick the step size
    if isinstance(solver, diffrax.Dopri5):
        dt0 = None

    batch_size = x_init.shape[0]
    sample_shape = x_init.shape[1:]  # e.g. (dim,) or (features, channel)
    flat_dim = math.prod(sample_shape)

    # Apply conditioning to initial samples
    if condition_mask is not None:
        x_init = x_init * (1 - condition_mask) + condition_value * condition_mask
        # Flatten mask for per-sample use
        condition_mask_flat = condition_mask.reshape(batch_size, flat_dim)
    else:
        condition_mask_flat = None

    def make_reverse_drift(cond_mask_i):
        """Create reverse drift closure for a single sample."""

        def reverse_drift(t, y_flat, args):
            # y_flat: (flat_dim,) -> reshape to sample_shape for model
            y = y_flat.reshape(sample_shape)
            # Add batch dim for model call; pass t as scalar (model handles expansion)
            y_batched = y[None, ...]  # (1, ...)
            t_batched = jnp.atleast_1d(t)[None, ...]  # (1, 1)
            score = score_model(obs=y_batched, t=t_batched, **args)
            score = jnp.squeeze(score, axis=0)  # remove batch dim

            # Broadcast t to spatial shape only for SDE coefficient computations
            t_broadcast = jnp.broadcast_to(t, y.shape)
            g_sq = sde.diffusion(t_broadcast) ** 2
            forward_drift = sde.drift(y, t_broadcast)
            result = forward_drift - 0.5 * g_sq * score.reshape(sample_shape)
            if cond_mask_i is not None:
                result = result * (1 - cond_mask_i.reshape(sample_shape))
            return result.reshape(flat_dim)

        return reverse_drift

    def sample_one(key_i, y0_flat, cond_mask_i):
        drift_fn = make_reverse_drift(cond_mask_i)

        terms = ODETerm(drift_fn)

        if return_intermediates:
            saveat = SaveAt(ts=jnp.linspace(t0, t1, n_steps + 1))
        else:
            saveat = SaveAt(t1=True)

        sol = diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0=dt0,
            y0=y0_flat,
            args=model_kwargs,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        return sol.ys  # (n_saves, flat_dim)

    # Flatten inputs for diffrax
    x_init_flat = x_init.reshape(batch_size, flat_dim)

    keys = jax.random.split(key, batch_size)

    if condition_mask_flat is not None:
        results = jax.vmap(sample_one)(keys, x_init_flat, condition_mask_flat)
    else:
        # Use None broadcast via vmap in_axes
        results = jax.vmap(sample_one, in_axes=(0, 0, None))(keys, x_init_flat, None)

    # results shape: (batch, n_saves, flat_dim)
    if return_intermediates:
        # (batch, n_steps+1, flat_dim) -> (n_steps+1, batch, *sample_shape)
        results = results.reshape(batch_size, n_steps + 1, *sample_shape)
        ndim = len(sample_shape)
        perm = (1, 0) + tuple(range(2, 2 + ndim))
        return jnp.transpose(results, perm)
    else:
        # (batch, 1, flat_dim) -> (batch, *sample_shape)
        return results.reshape(batch_size, *sample_shape)
