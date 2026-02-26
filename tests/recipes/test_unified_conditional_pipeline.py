import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np

from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel

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
    )


train_dataset = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

val_dataset = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
def test_unified_conditional_init(method):
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method,
    )
    assert isinstance(pipeline, ConditionalPipeline)
    assert hasattr(pipeline, "obs_ids")
    assert hasattr(pipeline, "cond_ids")
    assert hasattr(pipeline, "path")
    assert hasattr(pipeline, "loss_obj")


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
def test_unified_conditional_train_and_sample(method):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=method,
            ch_obs=2,
            ch_cond=2,
            training_config=training_config,
        )

        # Train 2 steps
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=False)
        pipeline._wrap_model()

        # Direct model evaluation
        batch_size = 3
        t = jnp.linspace(0, 1, batch_size)
        obs = jnp.zeros((batch_size, dim_obs, 2))
        cond = jnp.zeros((batch_size, dim_cond, 2))
        out = pipeline.model_wrapped(t, obs, pipeline.obs_ids, cond, pipeline.cond_ids)
        assert out.shape == (batch_size, dim_obs, 2)
        out_ema = pipeline.ema_model_wrapped(t, obs, pipeline.obs_ids, cond, pipeline.cond_ids)
        assert out_ema.shape == (batch_size, dim_obs, 2)

        # Sample
        cond = jnp.zeros((1, dim_cond, 2))
        sample = pipeline.sample(
            jax.random.PRNGKey(1), cond, nsamples=32, use_ema=False
        )
        assert sample.shape == (32, dim_obs, 2)

        sample_ema = pipeline.sample(
            jax.random.PRNGKey(1), cond, nsamples=32, use_ema=True
        )
        assert sample_ema.shape == (32, dim_obs, 2)


@pytest.mark.parametrize(
    "method",
    [
        FlowMatchingMethod(),
        DiffusionEDMMethod(),
        ScoreMatchingMethod(),
    ],
    ids=["flow", "diffusion", "score"],
)
def test_unified_conditional_loss_fn(method):
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method,
        ch_obs=2,
        ch_cond=2,
    )
    pipeline._wrap_model()

    loss_fn = pipeline.get_loss_fn()
    mock_batch = (jnp.zeros((32, dim_obs, 2)), jnp.zeros((32, dim_cond, 2)))
    loss = loss_fn(pipeline.model_wrapped, mock_batch, key=jax.random.PRNGKey(1))
    assert loss.shape == ()
