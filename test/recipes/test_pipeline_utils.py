import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

from gensbi.recipes.joint_pipeline import sample_condition_mask 

def test_sample_condition_mask():
    key = jax.random.PRNGKey(0)
    num_samples = 10
    theta_dim = 5
    x_dim = 3
    kind = "structured"
    condition_mask = sample_condition_mask(key, num_samples, theta_dim, x_dim, kind)
    assert condition_mask.shape == (num_samples, theta_dim + x_dim, 1)
    kind = "posterior"
    condition_mask = sample_condition_mask(key, num_samples, theta_dim, x_dim, kind)
    assert condition_mask.shape == (num_samples, theta_dim + x_dim, 1)
    kind = "likelihood"
    condition_mask = sample_condition_mask(key, num_samples, theta_dim, x_dim, kind)
    assert condition_mask.shape == (num_samples, theta_dim + x_dim, 1)
    kind = "joint"
    condition_mask = sample_condition_mask(key, num_samples, theta_dim, x_dim, kind)
    assert condition_mask.shape == (num_samples, theta_dim + x_dim, 1)
    return


