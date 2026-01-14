import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

from gensbi.recipes.joint_pipeline import sample_condition_mask

from gensbi.recipes.utils import init_ids_1d, init_ids_2d, init_ids_joint


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


def test_init_ids_1d():
    dim = 5
    ids = init_ids_1d(dim)
    assert ids.shape == (1, dim, 1)
    assert (ids[0, :, 0] == jnp.arange(dim)).all()

    ids = init_ids_1d(dim, semantic_id=1)
    assert ids.shape == (1, dim, 2)
    assert (ids[0, :, 0] == jnp.arange(dim)).all()
    assert (ids[0, :, 1] == 1).all()
    return


def test_init_ids_2d():
    dim = (6, 6)
    ids = init_ids_2d(dim)
    assert ids.shape == (1, (dim[0] / 2) * (dim[1] / 2), 3)
    return


def test_init_ids_joint():
    dim_obs = 3
    dim_cond = 4
    node_ids, obs_ids, cond_ids = init_ids_joint(dim_obs, dim_cond)
    assert node_ids.shape == (1, 7, 1)
    assert obs_ids.shape == (1, 3, 1)
    assert cond_ids.shape == (1, 4, 1)
    return
