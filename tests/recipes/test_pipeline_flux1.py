"""Integration tests for Flux1 model-specific pipelines."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.models import Flux1Params
from gensbi.recipes import Flux1FlowPipeline, Flux1DiffusionPipeline
from gensbi.recipes.flux1 import Flux1SMPipeline

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_test_helpers import run_model_pipeline_test, run_load_config_test, dim_obs, dim_cond


# --- Model params ---

params_flux_flow = Flux1Params(
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[2],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    theta=20,
    id_merge_mode="sum",
    id_embedding_strategy=("absolute", "absolute"),
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

params_flux_diff = Flux1Params(
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    val_emb_dim=4,
    id_emb_dim=4,
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    theta=20,
    id_merge_mode="concat",
    id_embedding_strategy=("absolute", "pos1d"),
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

params_flux_sm = Flux1Params(
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[2],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    theta=20,
    id_merge_mode="sum",
    id_embedding_strategy=("absolute", "absolute"),
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)


# --- Config paths ---

config_flow = "tests/recipes/configs/config_flow_flux.yaml"
config_diff = "tests/recipes/configs/config_diffusion_flux.yaml"
config_sm = "tests/recipes/configs/config_sm_flux.yaml"


# --- Tests ---

@pytest.mark.parametrize(
    "pipeline_cls, config_path",
    [
        (Flux1FlowPipeline, config_flow),
        (Flux1DiffusionPipeline, config_diff),
        (Flux1SMPipeline, config_sm),
    ],
)
def test_load_configs(pipeline_cls, config_path):
    run_load_config_test(pipeline_cls, config_path, is_conditional=True)


@pytest.mark.parametrize("pipeline_cls", [Flux1FlowPipeline, Flux1DiffusionPipeline, Flux1SMPipeline])
def test_defaults(pipeline_cls):
    default_params = pipeline_cls.get_default_params(1, 2, 1, 1)
    assert isinstance(default_params, Flux1Params)


@pytest.mark.parametrize(
    "pipeline_cls, params",
    [
        (Flux1FlowPipeline, params_flux_flow),
        (Flux1DiffusionPipeline, params_flux_diff),
        (Flux1SMPipeline, params_flux_sm),
    ],
)
@pytest.mark.slow
def test_model_pipeline(pipeline_cls, params):
    run_model_pipeline_test(pipeline_cls, params, is_conditional=True)
