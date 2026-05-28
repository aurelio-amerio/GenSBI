import jax
from jax import numpy as jnp
from jax import jit
from jax import Array
from typing import Callable, Optional, Any

from einops import repeat


def edm_sampler(
    sde: Any,
    model: Callable,
    x_1: Array,
    *,
    key: Array,
    return_intermediates: bool = False,
    n_steps: int = 18,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    method: str = "Heun",
    model_kwargs: dict = None,
) -> Array:
    """
    EDM sampler for diffusion models.

    **Time direction convention:**
    The EDM sampler operates in **σ-space** (noise scale), not a conventional
    time variable. It steps through a decreasing schedule ``σ_max → 0``, where
    large σ = noisy and σ=0 = clean data. This is different from both flow
    matching (``t: 0→1``, noise→data) and standard score matching
    (reverse SDE: ``t: T→eps``, noise→data).

    Parameters
    ----------
        sde: SDE scheduler object.
        model : Callable
            Model function.
        x_1 : Array
            Initial value.
        key : Array
            JAX random key.
        return_intermediates : bool
            Whether to return intermediate steps.
        n_steps : int
            Number of steps.
        S_churn : float
            Churn parameter.
        S_min : float
            Minimum S value.
        S_max : float
            Maximum S value.
        S_noise : float
            Noise scale.
        method : str
            Integration method ("Euler" or "Heun").
        model_kwargs : dict
            Additional model arguments.

    Returns
    -------
        Array
            Sampled output.
    """
    assert method in ["Euler", "Heun"], f"Unknown method: {method}"
    if model_kwargs is None:
        model_kwargs = {}

    # Time step discretization.
    step_indices = jnp.arange(n_steps)

    t_steps = sde.timesteps(step_indices, n_steps)
    t_steps = jnp.append(t_steps, 0)

    # Main sampling loop.
    x_next = x_1 * t_steps[0]

    def one_step(carry, i):
        x_next, key = carry
        key, subkey = jax.random.split(key)
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_curr = x_next

        # Increase noise temporarily.
        in_range = jnp.logical_and(t_cur >= S_min, t_cur <= S_max)
        gamma = jax.lax.cond(
            in_range,
            lambda: jnp.minimum(S_churn / n_steps, jnp.sqrt(2) - 1),
            lambda: 0.0,
        )
        t_hat = t_cur + gamma * t_cur  # sigma at the specific time step
        sqrt_arg = jnp.clip(t_hat**2 - t_cur**2, min=0, max=None)
        x_hat = x_curr + jnp.sqrt(sqrt_arg) * S_noise * jax.random.normal(
            subkey, x_curr.shape
        )

        # Euler step.
        denoised = sde.denoise(
            model, x_hat, t_hat[..., None], **model_kwargs
        )
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if method == "Heun":
            # Apply 2nd order correction.
            def apply_2nd_order_correction():  # Function for i < (n_steps - 1)
                denoised = sde.denoise(model, x_next, t_next[..., None], **model_kwargs)
                d_prime = (x_next - denoised) / t_next
                x_next_updated = x_hat + (t_next - t_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )
                return x_next_updated

            x_next = jax.lax.cond(
                i < (n_steps - 1), apply_2nd_order_correction, lambda: x_next
            )  # Apply 2nd order correction if i < (n_steps - 1)

        if return_intermediates:
            return (x_next, key), x_next
        else:
            return (x_next, key), ()

    i = jnp.arange(n_steps)

    carry, x_scan = jax.lax.scan(one_step, (x_next, key), i)
    if return_intermediates:
        return x_scan
    else:
        return carry[0]


def edm_ablation_sampler(
    sampling_scheduler,
    denoise_scheduler,
    model,
    x_1,
    *,
    key,
    return_intermediates=False,
    n_steps=18,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    method="Heun",
    model_kwargs=None,
):
    """Generalized ablation sampler for EDM diffusion models.

    Decouples the **sampling schedule** (time discretization, scaling) from
    the **preconditioning** (denoiser wrapper).  This allows sampling an
    EDM-trained model using VP or VE noise schedules without changing the
    model's internal preconditioning.

    Parameters
    ----------
    sampling_scheduler
        Scheduler that controls the sampling dynamics: ``sigma``, ``s``,
        ``sigma_deriv``, ``s_deriv``, ``sigma_inv``, and ``timesteps``.
    denoise_scheduler
        Scheduler that provides the ``denoise`` method (preconditioning:
        ``c_skip``, ``c_in``, ``c_out``, ``c_noise``).  This must match
        the scheduler used during training.
    model : Callable
        Model function (raw network, without preconditioning).
    x_1 : Array
        Initial latent noise.
    key : Array
        JAX random key.
    return_intermediates : bool
        Whether to return intermediate steps.
    n_steps : int
        Number of sampling steps.
    S_churn : float
        Stochasticity strength.
    S_min : float
        Minimum sigma for stochastic noise injection.
    S_max : float
        Maximum sigma for stochastic noise injection.
    S_noise : float
        Noise inflation factor.
    method : str
        Integration method (``"Euler"`` or ``"Heun"``).
    model_kwargs : dict
        Additional model arguments.

    Returns
    -------
    Array
        Sampled output.
    """

    assert method in ["Euler", "Heun"], f"Unknown method: {method}"
    if model_kwargs is None:
        model_kwargs = {}

    # Time step discretization.
    step_indices = jnp.arange(n_steps)

    sde = sampling_scheduler

    t_steps = sde.timesteps(step_indices, n_steps)
    t_steps = jnp.append(t_steps, 0)

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = x_1 * (sde.sigma(t_next) * sde.s(t_next))

    def one_step(carry, i):
        x_next, key = carry
        key, subkey = jax.random.split(key)
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_curr = x_next

        # Increase noise temporarily.
        sigma_cur = sde.sigma(t_cur)
        in_range = jnp.logical_and(sigma_cur >= S_min, sigma_cur <= S_max)

        gamma = jax.lax.cond(
            in_range,
            lambda: jnp.minimum(S_churn / n_steps, jnp.sqrt(2) - 1),
            lambda: 0.0,
        )
        t_hat = jnp.where(
            gamma > 0, sde.sigma_inv(sde.sigma(t_cur) + gamma * sde.sigma(t_cur)), t_cur
        )
        sqrt_arg = jnp.where(
            gamma > 0,
            jnp.clip(sde.sigma(t_hat) ** 2 - sde.sigma(t_cur) ** 2, min=0, max=None),
            0.0,
        )
        x_hat = sde.s(t_hat) / sde.s(t_cur) * x_curr + jnp.sqrt(sqrt_arg) * sde.s(
            t_hat
        ) * S_noise * jax.random.normal(subkey, x_curr.shape)

        # Euler step.
        h = t_next - t_hat
        denoised = denoise_scheduler.denoise(
            model,
            x_hat / sde.s(t_hat),
            sde.sigma(t_hat)[..., None],
            **model_kwargs,
        )
        d_cur = (
            sde.sigma_deriv(t_hat) / sde.sigma(t_hat)
            + sde.s_deriv(t_hat) / sde.s(t_hat)
        ) * x_hat - sde.sigma_deriv(t_hat) * sde.s(t_hat) / sde.sigma(t_hat) * denoised
        x_prime = x_hat + h * d_cur
        t_prime = t_next

        if method == "Heun":
            # Apply 2nd order correction.
            def apply_2nd_order_correction():  # Function for i < (n_steps - 1)
                denoised = denoise_scheduler.denoise(
                    model,
                    x_prime / sde.s(t_prime),
                    sde.sigma(t_prime)[..., None],
                    **model_kwargs,
                )
                d_prime = (
                    sde.sigma_deriv(t_prime) / sde.sigma(t_prime)
                    + sde.s_deriv(t_prime) / sde.s(t_prime)
                ) * x_prime - sde.sigma_deriv(t_prime) * sde.s(t_prime) / sde.sigma(
                    t_prime
                ) * denoised
                x_next = x_hat + h * (
                    0.5 * d_cur + 0.5 * d_prime
                )
                return x_next

            x_next = jax.lax.cond(
                i < (n_steps - 1), apply_2nd_order_correction, lambda: x_prime
            )  # Apply 2nd order correction if i < (n_steps - 1)
        else:
            x_next = x_prime

        if return_intermediates:
            return (x_next, key), x_next
        else:
            return (x_next, key), ()

    i = jnp.arange(n_steps)

    carry, x_scan = jax.lax.scan(one_step, (x_next, key), i)
    if return_intermediates:
        return x_scan
    else:
        return carry[0]
