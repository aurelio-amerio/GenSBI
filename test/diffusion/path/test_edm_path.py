import os
os.environ['JAX_PLATFORMS'] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler


def test_edm_path_initialization():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    assert isinstance(path, EDMPath)
    assert path.scheduler is scheduler


def test_edm_path_sample_sigma_shape():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    key = jax.random.PRNGKey(0)
    batch_size = 10
    sigma = path.sample_sigma(key, batch_size)
    assert sigma.shape == (batch_size,)


def test_edm_path_sample_prior_shape():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    key = jax.random.PRNGKey(0)
    shape = (5, 2)
    prior = path.sample_prior(key, shape)
    assert prior.shape == shape


def test_edm_path_get_loss_fn_callable():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    loss_fn = path.get_loss_fn()
    assert callable(loss_fn)


def test_edm_path_sample():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    key = jax.random.PRNGKey(0)
    x_1 = jnp.ones((2, 3))
    sigma = jnp.ones((2, 1))
    sample = path.sample(key, x_1, sigma)
    assert hasattr(sample, 'x_1')
    assert hasattr(sample, 'sigma')
    assert hasattr(sample, 'x_t')
    assert sample.x_1.shape == x_1.shape
    assert sample.sigma.shape == sigma.shape
    assert sample.x_t.shape == x_1.shape
