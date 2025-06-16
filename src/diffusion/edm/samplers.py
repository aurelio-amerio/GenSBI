import jax
from jax import numpy as jnp
from jax import jit

# from diffrax import (
#     diffeqsolve,
#     ControlTerm,
#     MultiTerm,
#     ODETerm,
#     VirtualBrownianTree,
#     UnsafeBrownianPath,
#     DirectAdjoint,
#     Euler,
# )

# TODO still need to test
def edm_sampler(
    sde, model, x, *,
    key,
    condition_mask = None,
    condition_value = None,
    n_steps=18,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    method="heun",
    model_kwargs={},
):

    assert method in ["euler", "heun"], f"Unknown method: {method}"
    if condition_mask is not None:
            assert (
                condition_value is not None
            ), "Condition value must be provided if condition mask is provided"
    else:
        condition_mask = 0
        condition_value = 0

    # Time step discretization.
    step_indices = jnp.arange(n_steps)

    t_steps = sde.timesteps(step_indices, n_steps)
    t_steps = jnp.append(t_steps, 0)

    # Main sampling loop.
    x_next = x * t_steps[0]

    def one_step(carry, i):
        x_next, key = carry
        key, subkey = jax.random.split(key)
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        x_curr = x_next

        # Increase noise temporarily.
        in_range = jnp.logical_and(t_cur >= S_min, t_cur <= S_max)
        # print(in_range)
        gamma = jax.lax.cond(in_range, lambda: jnp.minimum(S_churn / n_steps, jnp.sqrt(2) - 1), lambda: 0.0)
        t_hat = t_cur + gamma * t_cur # sigma at the specific time step
        sqrt_arg = jnp.clip(t_hat ** 2 - t_cur ** 2, a_min=0, a_max=None)
        x_hat = x_curr + jnp.sqrt(sqrt_arg) * S_noise * jax.random.normal(subkey, x_curr.shape)
        x_hat = x_hat * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.
        # Euler step.
        denoised = sde.denoise(model, x_hat, jnp.broadcast_to(t_hat, (x_hat.shape[0],1)), **model_kwargs)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        x_next = x_next * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.

        if method == "heun":
            # Apply 2nd order correction.        
            def apply_2nd_order_correction():  # Function for i < (n_steps - 1)
                denoised = sde.denoise(model, x_next, jnp.broadcast_to(t_next, (x_next.shape[0],1)), **model_kwargs)
                d_prime = (x_next - denoised) / t_next
                x_next_updated = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)  # Store in a new variable
                x_next_updated = x_next_updated * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.
                return x_next_updated  # Return the updated x_next
            
            x_next = jax.lax.cond(i < (n_steps - 1), apply_2nd_order_correction, lambda: x_next)  # Apply 2nd order correction if i < (n_steps - 1)
        
        return (x_next, key), ()
    
    i = jnp.arange(n_steps)
    # return one_step, x_next
        
    carry, _ = jax.lax.scan(one_step, (x_next, key), i)
    return carry[0]


def edm_ablation_sampler(
    sde, model, x, *,
    key,
    condition_mask = None,
    condition_value = None,
    n_steps=18,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    method="heun",
    model_kwargs={},
):

    assert method in ["euler", "heun"], f"Unknown method: {method}"
    if condition_mask is not None:
            assert (
                condition_value is not None
            ), "Condition value must be provided if condition mask is provided"
    else:
        condition_mask = 0
        condition_value = 0

    # Time step discretization.
    step_indices = jnp.arange(n_steps)

    t_steps = sde.timesteps(step_indices, n_steps)
    t_steps = jnp.append(t_steps, 0)

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = x * (sde.sigma(t_next) * sde.s(t_next))

    def one_step(carry, i):
        x_next, key = carry
        key, subkey = jax.random.split(key)
        t_cur = t_steps[i]
        t_next = t_steps[i+1]
        x_curr = x_next

        # Increase noise temporarily.
        in_range = jnp.logical_and(t_cur >= S_min, t_cur <= S_max)

        gamma = jax.lax.cond(in_range, lambda: jnp.minimum(S_churn / n_steps, jnp.sqrt(2) - 1), lambda: 0.0)
        t_hat = sde.sigma_inv(sde.sigma(t_cur) + gamma * sde.sigma(t_cur)) # sigma at the specific time step
        sqrt_arg = jnp.clip(sde.sigma(t_hat) ** 2 - sde.sigma(t_cur) ** 2, a_min=0, a_max=None)
        x_hat = sde.s(t_hat) / sde.s(t_cur)*x_curr + jnp.sqrt(sqrt_arg) * sde.s(t_hat)*S_noise * jax.random.normal(subkey, x_curr.shape)
        x_hat = x_hat * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.
        # Euler step.
        h = t_next - t_hat
        denoised = sde.denoise(model, x_hat/sde.s(t_hat), jnp.broadcast_to(sde.sigma(t_hat), (x_hat.shape[0],1)), **model_kwargs)
        d_cur = (sde.sigma_deriv(t_hat) / sde.sigma(t_hat) + sde.s_deriv(t_hat) / sde.s(t_hat)) * x_hat - sde.sigma_deriv(t_hat) * sde.s(t_hat) / sde.sigma(t_hat) * denoised
        x_prime = x_hat + h * d_cur
        t_prime = t_next
        x_prime = x_prime * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.

        if method == "heun":
            # Apply 2nd order correction.        
            def apply_2nd_order_correction():  # Function for i < (n_steps - 1)
                denoised = sde.denoise(model, x_prime/sde.s(t_prime), jnp.broadcast_to(sde.sigma(t_prime), (x_prime.shape[0],1)), **model_kwargs)
                d_prime = (sde.sigma_deriv(t_prime) / sde.sigma(t_prime) + sde.s_deriv(t_prime) / sde.s(t_prime)) * x_prime - sde.sigma_deriv(t_prime) * sde.s(t_prime) / sde.sigma(t_prime) * denoised
                x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)  # Store in a new variable
                x_next = x_next * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.
                return x_next  # Return the updated x_next
            
            x_next = jax.lax.cond(i < (n_steps - 1), apply_2nd_order_correction, lambda: x_prime)  # Apply 2nd order correction if i < (n_steps - 1)
        else:
            x_next = x_prime

        return (x_next, key), ()
    
    i = jnp.arange(n_steps)
    # return one_step, x_next
        
    carry, _ = jax.lax.scan(one_step, (x_next, key), i)
    return carry[0]


def sampler(
    sde, model, x, *,
    key,
    **kwargs,
):
    # get conditional mask from kwargs, or set it to none. pop it from kwargs
    condition_mask = kwargs.pop("condition_mask", None)
    condition_value = kwargs.pop("condition_value", None)

    if sde.name in ["VP", "VE"]:
        sampler_ = edm_ablation_sampler
    elif sde.name == "EDM":
        sampler_ = edm_sampler
    
    return sampler_(sde, model, x, key=key, condition_mask=condition_mask, condition_value=condition_value, **kwargs)
