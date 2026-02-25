import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

from gensbi.recipes import (
    ConditionalPipeline,
    JointPipeline,
    UnconditionalPipeline,
)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockUnconditionalModel

from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

from gensbi.models import Simformer, SimformerParams, Flux1, Flux1Params

import grain
import numpy as np


nsamples = 1000
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond


theta = jax.random.normal(key, (nsamples, dim_obs, 2))
x = jax.random.normal(key, (nsamples, dim_cond, 2))

data = jnp.concatenate([theta, x], axis=1)


def split_obs_cond(data):
    return (
        data[:, :dim_obs],
        data[:, dim_obs:],
    )  # assuming first dim_obs are obs, last dim_cond are cond


train_dataset_joint = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

val_dataset_joint = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

train_dataset_cond = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

val_dataset_cond = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

# Simformer model for joint/unconditional tests
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
    mlp_ratio=3,
    qkv_features=4,
    num_hidden_layers=1,
)

model_joint = Simformer(params_simf)

# Flux1 model for conditional tests
params = Flux1Params(
    in_channels=2,
    vec_in_dim=None,
    context_in_dim=2,
    mlp_ratio=1,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[
        2,
    ],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    theta=20,
    id_merge_mode="sum",
    id_embedding_strategy=("absolute", "absolute"),
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model_conditional = Flux1(params)


# Parametrize by (method, is_conditional) to test all unified pipeline combinations
@pytest.mark.parametrize(
    "method, is_conditional",
    [
        (FlowMatchingMethod(), True),
        (DiffusionEDMMethod(), True),
        (ScoreMatchingMethod(), True),
        (FlowMatchingMethod(), False),
        (DiffusionEDMMethod(), False),
        (ScoreMatchingMethod(), False),
    ],
)
@pytest.mark.slow
def test_model_general_conditional(method, is_conditional):

    if is_conditional:
        model = model_conditional
        train_dataset = train_dataset_cond
        val_dataset = val_dataset_cond
        PipelineCls = ConditionalPipeline
    else:
        model = model_joint
        train_dataset = train_dataset_joint
        val_dataset = val_dataset_joint
        PipelineCls = JointPipeline

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = PipelineCls.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        if is_conditional:
            default_pipeline = PipelineCls(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dim_obs=dim_obs,
                dim_cond=dim_cond,
                method=method,
                ch_obs=2,
                ch_cond=2,
            )
        else:
            default_pipeline = PipelineCls(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dim_obs=dim_obs,
                dim_cond=dim_cond,
                method=method,
                ch_obs=2,
            )

        assert isinstance(default_pipeline, PipelineCls)

        if is_conditional:
            pipeline = PipelineCls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                method=method,
                ch_obs=2,
                ch_cond=2,
                training_config=training_config,
            )
        else:
            pipeline = PipelineCls(
                model,
                train_dataset,
                val_dataset,
                dim_obs,
                dim_cond,
                method=method,
                ch_obs=2,
                training_config=training_config,
            )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_obs, 2))
        cond = jnp.ones((batch_size, dim_cond, 2))

        obs_ids = pipeline.obs_ids
        cond_ids = pipeline.cond_ids

        # try training the model
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        # wrap the model
        pipeline._wrap_model()

        # try evaluating the model, and save the result
        out = pipeline.model_wrapped(t, obs, obs_ids, cond, cond_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids, cond, cond_ids)
        assert out.shape == (
            batch_size,
            dim_obs,
            2,
        ), f"Expected shape {(batch_size, dim_obs, 2)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_obs,
            2,
        ), f"Expected shape {(batch_size, dim_obs, 2)}, got {out_ema.shape}"


########


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
)
@pytest.mark.slow
def test_model_general_unconditional(method):

    train_dataset = train_dataset_joint
    val_dataset = val_dataset_joint

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        model = MockUnconditionalModel()

        default_pipeline = UnconditionalPipeline(
            model, train_dataset, val_dataset, dim_joint, method=method
        )

        assert isinstance(default_pipeline, UnconditionalPipeline)

        model2 = MockUnconditionalModel()
        pipeline = UnconditionalPipeline(
            model2,
            train_dataset,
            val_dataset,
            dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.ones((batch_size, dim_joint, 2))

        obs_ids = pipeline.obs_ids

        # try training the model
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=True)
        # wrap the model
        pipeline._wrap_model()

        # try evaluating the model, and save the result
        out = pipeline.model_wrapped(t, obs, obs_ids)
        out_ema = pipeline.ema_model_wrapped(t, obs, obs_ids)
        assert out.shape == (
            batch_size,
            dim_joint,
            2,
        ), f"Expected shape {(batch_size, dim_joint, 2)}, got {out.shape}"
        assert out_ema.shape == (
            batch_size,
            dim_joint,
            2,
        ), f"Expected shape {(batch_size, dim_joint, 2)}, got {out_ema.shape}"
