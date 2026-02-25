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
    ids, dim_ = init_ids_1d(dim)
    assert ids.shape == (1, dim, 1)
    assert (ids[0, :, 0] == jnp.arange(dim)).all()
    assert dim == dim_

    ids, _ = init_ids_1d(dim, semantic_id=1)
    assert ids.shape == (1, dim, 2)
    assert (ids[0, :, 0] == jnp.arange(dim)).all()
    assert (ids[0, :, 1] == 1).all()
    return


def test_init_ids_2d():
    dim = (6, 6)
    ids, dim_ = init_ids_2d(dim)
    assert ids.shape == (1, (dim[0] / 2) * (dim[1] / 2), 3)
    assert (dim[0] // 2) * (dim[1] // 2) == dim_
    return


def test_init_ids_joint():
    dim_obs = 3
    dim_cond = 4
    node_ids, obs_ids, cond_ids = init_ids_joint(dim_obs, dim_cond)
    assert node_ids.shape == (1, 7, 1)
    assert obs_ids.shape == (1, 3, 1)
    assert cond_ids.shape == (1, 4, 1)


# ---------------------------------------------------------------------------
# Tests for uncovered lines in recipes/utils.py
# ---------------------------------------------------------------------------

from gensbi.recipes.utils import (
    patchify_2d,
    _resolve_embedding_ids,
    build_edm_path,
    build_sm_path,
    parse_training_config,
)


def test_patchify_2d():
    """patchify_2d rearranges (B, H, W, C) -> (B, H*W/4, C*4)."""
    x = jnp.ones((2, 4, 6, 3))
    out = patchify_2d(x)
    # ph=pw=2, so patches are 2x2, H/2=2, W/2=3 -> (2, 2*3, 3*4)
    assert out.shape == (2, 6, 12)


def test_resolve_embedding_ids_unknown_strategy():
    """Unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        _resolve_embedding_ids("invalid_strategy", 5, semantic_id=0)


def test_build_edm_path_invalid_sde():
    """Invalid SDE type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sde type"):
        build_edm_path("INVALID", {})


def test_build_sm_path_invalid_sde():
    """Invalid SDE type raises ValueError."""
    with pytest.raises(ValueError, match="sde_type must be"):
        build_sm_path("INVALID", {})


def test_parse_training_config_ema_decay_in_optimizer():
    """ema_decay in optimizer section overrides default (backward compat)."""
    import tempfile
    import yaml

    config = {
        "training": {"nsteps": 100},
        "optimizer": {"ema_decay": 0.123, "max_lr": 1e-3},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()
        result = parse_training_config(f.name)

    assert result["ema_decay"] == 0.123

