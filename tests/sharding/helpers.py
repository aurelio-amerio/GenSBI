"""Shared helpers for the multi-device sharding tests."""

import importlib.util
import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi import sharding as sh
from gensbi.models import Flux1JointParams, Flux1Params, SimformerParams
from gensbi.models.embedding.embedding import PEMatrix
from gensbi.normalizing_flows.bijections.base import Mask
from gensbi.recipes.pipeline import AbstractPipeline

# ``tests`` is a package (see tests/__init__.py), so this import works from the
# repo root under both ``-p no:xdist`` and ``-n 2``.
from tests.recipes.model_test_helpers import (  # noqa: F401
    dim_cond,
    dim_joint,
    dim_obs,
    train_dataset_cond,
    train_dataset_joint,
    val_dataset_cond,
    val_dataset_joint,
)

requires_8_devices = pytest.mark.skipif(
    jax.device_count() < 8,
    reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8",
)


def has_healswin_sharding() -> bool:
    """True if the installed heal_swin_nnx ships a ``sharding`` module."""
    try:
        return importlib.util.find_spec("heal_swin_nnx.sharding") is not None
    except ModuleNotFoundError:
        return False


def _check_hidden(hidden_size: int) -> int:
    assert hidden_size % 8 == 0, f"hidden size {hidden_size} not divisible by 8"
    return hidden_size


def small_flux1_params() -> Flux1Params:
    """Tiny Flux1 config with residual width 16."""
    params = Flux1Params(
        in_channels=2,
        vec_in_dim=None,
        context_in_dim=2,
        mlp_ratio=1,
        num_heads=2,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[8],
        qkv_bias=True,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        theta=20,
        id_merge_mode="sum",
        id_embedding_strategy=("absolute", "absolute"),
        rngs=nnx.Rngs(0),
        param_dtype=jnp.float32,
    )
    _check_hidden(params.hidden_size)
    return params


def small_simformer_params() -> SimformerParams:
    """Tiny Simformer config with residual width 16."""
    params = SimformerParams(
        rngs=nnx.Rngs(0),
        in_channels=2,
        val_emb_dim=8,
        id_emb_dim=4,
        cond_emb_dim=4,
        dim_joint=dim_joint,
        fourier_features=16,
        num_heads=2,
        depth=1,
        mlp_ratio=1,
        qkv_features=8,
        num_hidden_layers=1,
    )
    _check_hidden(params.val_emb_dim + params.id_emb_dim + params.cond_emb_dim)
    return params


def small_flux1joint_params(id_merge_mode: str = "sum") -> Flux1JointParams:
    """Tiny Flux1Joint config with residual width 16 (32 for ``id_merge_mode="concat"``)."""
    params = Flux1JointParams(
        in_channels=2,
        vec_in_dim=None,
        mlp_ratio=1,
        num_heads=2,
        depth_single_blocks=1,
        val_emb_dim=8,
        cond_emb_dim=4,
        id_emb_dim=4,
        qkv_bias=True,
        rngs=nnx.Rngs(0),
        dim_joint=dim_joint,
        id_merge_mode=id_merge_mode,
        id_embedding_strategy="absolute",
        guidance_embed=False,
        param_dtype=jnp.float32,
    )
    _check_hidden(params.hidden_size)
    return params


def test_training_config(tmp_path, nsteps=20) -> dict:
    """Default training config shrunk for fast tests."""
    cfg = AbstractPipeline.get_default_training_config()
    cfg["nsteps"] = nsteps
    cfg["val_every"] = 5
    cfg["warmup_steps"] = 2
    cfg["early_stopping"] = False
    cfg["checkpoint_dir"] = str(tmp_path / "ckpt")
    return cfg


# Not a test, despite the name (pytest would otherwise collect it on import).
test_training_config.__test__ = False


def assert_every_param_annotated(model):
    """Every ``nnx.Param`` carries a valid logical-axis annotation; constants carry none."""
    for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        meta = var.get_metadata()
        name = "/".join(map(str, path))
        assert "out_sharding" in meta, f"{name} has no annotation"
        assert meta["eager_sharding"] is False, name
        names = meta["out_sharding"]
        assert len(names) == var[...].ndim, f"{name}: rank {var[...].ndim} vs {names}"
        assert set(n for n in names if n is not None) <= set(sh.LOGICAL_AXES), name
        used = [n for n in names if n is not None]
        assert len(used) == len(set(used)), f"{name}: repeated logical name in {names}"
    for path, var in nnx.to_flat_state(nnx.state(model, nnx.Any(Mask, PEMatrix))):
        assert "out_sharding" not in var.get_metadata(), path
