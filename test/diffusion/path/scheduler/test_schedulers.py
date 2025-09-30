import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.scheduler import EDMScheduler,  VEScheduler, VPScheduler
import jax.numpy as jnp
import jax

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_name(scheduler_cls):
    scheduler = scheduler_cls()
    # assert the name is set
    assert isinstance(scheduler.name, str), f"scheduler.name is not a string, got type {type(scheduler.name)} and value {scheduler.name}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_time_schedule(scheduler_cls):
    scheduler = scheduler_cls()
    u = jnp.array([1e-2, 0.5, 1.0])
    t = scheduler.time_schedule(u)
    assert t.shape == u.shape, f"time_schedule: expected shape {u.shape}, got {t.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_timesteps(scheduler_cls):
    scheduler = scheduler_cls()
    i = jnp.arange(5)
    N = 5
    t = scheduler.timesteps(i, N)
    assert t.shape == i.shape, f"timesteps: expected shape {i.shape}, got {t.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sigma_and_inv(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    sigma = scheduler.sigma(t)
    t_inv = scheduler.sigma_inv(sigma)
    assert sigma.shape == t.shape, f"sigma: expected shape {t.shape}, got {sigma.shape}"
    assert t_inv.shape == t.shape, f"sigma_inv: expected shape {t.shape}, got {t_inv.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sigma_deriv(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    deriv = scheduler.sigma_deriv(t)
    assert deriv.shape == t.shape, f"sigma_deriv: expected shape {t.shape}, got {deriv.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_s_and_deriv(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    s = scheduler.s(t)
    s_deriv = scheduler.s_deriv(t)
    assert s.shape == t.shape, f"s: expected shape {t.shape}, got {s.shape}"
    assert s_deriv.shape == t.shape, f"s_deriv: expected shape {t.shape}, got {s_deriv.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_c_skip(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    c_skip = scheduler.c_skip(t)
    assert c_skip.shape == t.shape, f"c_skip: expected shape {t.shape}, got {c_skip.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_c_out(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    c_out = scheduler.c_out(t)
    assert c_out.shape == t.shape, f"c_out: expected shape {t.shape}, got {c_out.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_c_in(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    c_in = scheduler.c_in(t)
    assert c_in.shape == t.shape, f"c_in: expected shape {t.shape}, got {c_in.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_c_noise(scheduler_cls):
    scheduler = scheduler_cls()
    t = jnp.array([0.0, 1.0])
    c_noise = scheduler.c_noise(t)
    assert c_noise.shape == t.shape, f"c_noise: expected shape {t.shape}, got {c_noise.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sample_sigma(scheduler_cls):
    scheduler = scheduler_cls()
    key = jax.random.PRNGKey(0)
    shape_ = (2, 3)
    sigma = scheduler.sample_sigma(key, shape_)
    assert sigma.shape == shape_, f"sample_sigma: expected shape {shape_}, got {sigma.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sample_noise(scheduler_cls):
    scheduler = scheduler_cls()
    key = jax.random.PRNGKey(0)
    shape_ = (2, 3)
    sigma = jnp.array([0.1, 0.2])[..., None]
    noise = scheduler.sample_noise(key, shape_, sigma)
    assert noise.shape == shape_, f"sample_noise: expected shape {shape_}, got {noise.shape}, sigma shape: {sigma.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sample_prior(scheduler_cls):
    scheduler = scheduler_cls()
    key = jax.random.PRNGKey(0)
    shape_ = (2, 3)
    prior = scheduler.sample_prior(key, shape_)
    assert prior.shape == shape_, f"sample_prior: expected shape {shape_}, got {prior.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_loss_weight(scheduler_cls):   
    scheduler = scheduler_cls()
    sigma = jnp.array([0.1, 0.2])[..., None]
    weight = scheduler.loss_weight(sigma)
    assert weight.shape == sigma.shape, f"loss_weight: expected shape {sigma.shape}, got {weight.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_f(scheduler_cls):
    scheduler = scheduler_cls()
    x = jnp.ones((2, 3))
    t = jnp.array([0.0, 1.0])[..., None]
    f_out = scheduler.f(x, t)
    assert f_out.shape == x.shape, f"f: expected shape {x.shape}, got {f_out.shape}, t shape: {t.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_g(scheduler_cls):
    scheduler = scheduler_cls()
    x = jnp.ones((2, 3))
    t = jnp.array([0.0, 1.0])[..., None]
    g_out = scheduler.g(x, t)
    assert g_out.shape == t.shape, f"g: expected shape {t.shape}, got {g_out.shape}."

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_denoise(scheduler_cls):
    scheduler = scheduler_cls()

    def dummy_model(obs, t, *args, **kwargs):
        return jnp.zeros_like(obs)

    x = jnp.ones((2, 3))
    t = jnp.array([0.0, 1.0])[..., None]
    sigma = scheduler.sigma(t)
    denoised = scheduler.denoise(dummy_model, x, sigma)
    assert denoised.shape == x.shape, f"denoise: expected shape {x.shape}, got {denoised.shape}, sigma shape: {sigma.shape}"

@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_loss_fn(scheduler_cls):
    scheduler = scheduler_cls()

    def dummy_model(obs, t, *args, **kwargs):
        return jnp.zeros_like(obs)

    x = jnp.ones((2, 3))
    t = jnp.array([0.0, 1.0])[..., None]
    sigma = scheduler.sigma(t)
    loss_fn = scheduler.get_loss_fn()

    batch = (x, x, sigma)

    # (x_1, x_t, sigma) = batch

    loss = loss_fn(dummy_model, batch, None)
    # make sure the loss is a scalar
    assert loss.shape == (), f"loss_fn: expected shape (), got {loss.shape}"

    loss_mask = jnp.ones_like(x, dtype=bool)
    loss = loss_fn(dummy_model, batch, loss_mask)
    assert loss.shape == (), f"loss_fn: expected shape (), got {loss.shape}"

