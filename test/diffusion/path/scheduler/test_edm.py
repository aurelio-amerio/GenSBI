import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.scheduler import EDMScheduler
import jax.numpy as jnp


def test_edm_scheduler_initialization():
    scheduler = EDMScheduler()
    assert isinstance(scheduler, EDMScheduler)


def test_edm_scheduler_sigma_range():
    scheduler = EDMScheduler()
    sigma_min, sigma_max = scheduler.sigma_min, scheduler.sigma_max
    assert sigma_min < sigma_max


def test_edm_scheduler_timesteps():
    scheduler = EDMScheduler()
    i = jnp.arange(5)
    N = 5
    t = scheduler.timesteps(i, N)
    assert t.shape == i.shape


def test_edm_scheduler_sigma_and_inv():
    scheduler = EDMScheduler()
    t = jnp.array([0.0, 1.0])
    sigma = scheduler.sigma(t)
    t_inv = scheduler.sigma_inv(sigma)
    assert sigma.shape == t.shape
    assert t_inv.shape == t.shape


def test_edm_scheduler_sigma_deriv():
    scheduler = EDMScheduler()
    t = jnp.array([0.0, 1.0])
    deriv = scheduler.sigma_deriv(t)
    assert deriv.shape == t.shape

# Add more tests for s, s_deriv, and loss_weight if needed
