import os
os.environ['JAX_PLATFORMS'] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.diffusion.path.scheduler import VEScheduler, VPScheduler


def test_ve_scheduler_initialization():
    scheduler = VEScheduler()
    assert isinstance(scheduler, VEScheduler)
    assert scheduler.sigma_min < scheduler.sigma_max


def test_ve_scheduler_time_schedule_and_sigma():
    scheduler = VEScheduler()
    u = jnp.array([0.0, 0.5, 1.0])
    t = scheduler.time_schedule(u)
    sigma = scheduler.sigma(t)
    assert t.shape == u.shape
    assert sigma.shape == u.shape
    assert jnp.all(sigma >= 0)


def test_ve_scheduler_sample_sigma():
    scheduler = VEScheduler()
    key = jax.random.PRNGKey(0)
    shape = (5,)
    sigma = scheduler.sample_sigma(key, shape)
    assert sigma.shape == shape
    assert jnp.all(sigma >= scheduler.sigma_min)
    assert jnp.all(sigma <= scheduler.sigma_max)


def test_vp_scheduler_initialization():
    scheduler = VPScheduler()
    assert isinstance(scheduler, VPScheduler)
    assert scheduler.beta_min < scheduler.beta_max


def test_vp_scheduler_time_schedule_and_sigma():
    scheduler = VPScheduler()
    u = jnp.array([0.0, 0.5, 1.0])
    t = scheduler.time_schedule(u)
    sigma = scheduler.sigma(t)
    assert t.shape == u.shape
    assert sigma.shape == u.shape
    assert jnp.all(sigma >= 0)


def test_vp_scheduler_sample_sigma():
    scheduler = VPScheduler()
    key = jax.random.PRNGKey(0)
    shape = (5,)
    sigma = scheduler.sample_sigma(key, shape)
    assert sigma.shape == shape
    assert jnp.all(sigma >= 0)
