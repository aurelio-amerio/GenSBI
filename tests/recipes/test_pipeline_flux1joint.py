"""Integration tests for Flux1Joint model-specific pipelines."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.models import Flux1JointParams
from gensbi.recipes import Flux1JointFlowPipeline, Flux1JointDiffusionPipeline
from gensbi.recipes.flux1joint import Flux1JointSMPipeline

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_test_helpers import run_model_pipeline_test, run_load_config_test, dim_joint


# --- Model params ---

params_flux1joint_flow = Flux1JointParams(
    in_channels=2,
    vec_in_dim=None,
    mlp_ratio=1,
    num_heads=2,
    depth_single_blocks=2,
    val_emb_dim=4,
    cond_emb_dim=2,
    id_emb_dim=4,
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    dim_joint=dim_joint,
    id_merge_mode="sum",
    id_embedding_strategy="absolute",
    guidance_embed=False,
    param_dtype=jnp.float32,
)

params_flux1joint_diff = Flux1JointParams(
    in_channels=2,
    vec_in_dim=None,
    mlp_ratio=1,
    num_heads=2,
    depth_single_blocks=2,
    val_emb_dim=4,
    cond_emb_dim=2,
    id_emb_dim=4,
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    dim_joint=dim_joint,
    id_merge_mode="concat",
    id_embedding_strategy="absolute",
    guidance_embed=False,
    param_dtype=jnp.float32,
)

params_flux1joint_sm = Flux1JointParams(
    in_channels=2,
    vec_in_dim=None,
    mlp_ratio=1,
    num_heads=2,
    depth_single_blocks=2,
    val_emb_dim=4,
    cond_emb_dim=2,
    id_emb_dim=4,
    qkv_bias=True,
    rngs=nnx.Rngs(0),
    dim_joint=dim_joint,
    id_merge_mode="sum",
    id_embedding_strategy="absolute",
    guidance_embed=False,
    param_dtype=jnp.float32,
)


# --- Config paths ---

config_flow = "tests/recipes/configs/config_flow_flux1joint.yaml"
config_diff = "tests/recipes/configs/config_diffusion_flux1joint.yaml"
config_sm = "tests/recipes/configs/config_sm_flux1joint.yaml"


# --- Tests ---

@pytest.mark.parametrize(
    "pipeline_cls, config_path",
    [
        (Flux1JointFlowPipeline, config_flow),
        (Flux1JointDiffusionPipeline, config_diff),
        (Flux1JointSMPipeline, config_sm),
    ],
)
def test_load_configs(pipeline_cls, config_path):
    run_load_config_test(pipeline_cls, config_path, is_conditional=False)


@pytest.mark.parametrize("pipeline_cls", [Flux1JointFlowPipeline, Flux1JointDiffusionPipeline, Flux1JointSMPipeline])
def test_defaults(pipeline_cls):
    default_params = pipeline_cls.get_default_params(3, 1)
    assert isinstance(default_params, Flux1JointParams)


@pytest.mark.parametrize(
    "pipeline_cls, params",
    [
        (Flux1JointFlowPipeline, params_flux1joint_flow),
        (Flux1JointDiffusionPipeline, params_flux1joint_diff),
        (Flux1JointSMPipeline, params_flux1joint_sm),
    ],
)
@pytest.mark.slow
def test_model_pipeline(pipeline_cls, params):
    run_model_pipeline_test(pipeline_cls, params, is_conditional=False)
