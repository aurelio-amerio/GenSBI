"""Integration tests for Simformer model-specific pipelines."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.models import SimformerParams
from gensbi.recipes import SimformerFlowPipeline, SimformerDiffusionPipeline
from gensbi.recipes.simformer import SimformerSMPipeline

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_test_helpers import run_model_pipeline_test, run_load_config_test, dim_joint


# --- Model params ---

params_simf = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=2,
    val_emb_dim=2,
    id_emb_dim=2,
    cond_emb_dim=2,
    dim_joint=dim_joint,
    fourier_features=32,
    num_heads=2,
    depth=1,
    mlp_ratio=1,
    qkv_features=4,
    num_hidden_layers=1,
)


# --- Config paths ---

config_flow = "tests/recipes/configs/config_flow_simformer.yaml"
config_diff = "tests/recipes/configs/config_diffusion_simformer.yaml"
config_sm = "tests/recipes/configs/config_sm_simformer.yaml"


# --- Tests ---

@pytest.mark.parametrize(
    "pipeline_cls, config_path",
    [
        (SimformerFlowPipeline, config_flow),
        (SimformerDiffusionPipeline, config_diff),
        (SimformerSMPipeline, config_sm),
    ],
)
def test_load_configs(pipeline_cls, config_path):
    run_load_config_test(pipeline_cls, config_path, is_conditional=False)


@pytest.mark.parametrize("pipeline_cls", [SimformerFlowPipeline, SimformerDiffusionPipeline, SimformerSMPipeline])
def test_defaults(pipeline_cls):
    default_params = pipeline_cls.get_default_params(2, 1)
    assert isinstance(default_params, SimformerParams)


@pytest.mark.parametrize(
    "pipeline_cls, params",
    [
        (SimformerFlowPipeline, params_simf),
        (SimformerDiffusionPipeline, params_simf),
        (SimformerSMPipeline, params_simf),
    ],
)
@pytest.mark.slow
def test_model_pipeline(pipeline_cls, params):
    run_model_pipeline_test(pipeline_cls, params, is_conditional=False)
