import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest
import tempfile

import grain
import numpy as np

from gensbi.recipes import UnconditionalPipeline
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockUnconditionalModel

nsamples = 1000
key = jax.random.PRNGKey(0)

dim_joint = 9

theta = jax.random.normal(key, (nsamples, 2, 2))
x = jax.random.normal(key, (nsamples, 7, 2))
data = jnp.concatenate([theta, x], axis=1)

train_dataset = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
)

val_dataset = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
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
def test_unified_unconditional_init(method):
    pipeline = UnconditionalPipeline(
        model=MockUnconditionalModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_joint,
        method=method,
    )
    assert isinstance(pipeline, UnconditionalPipeline)
    assert hasattr(pipeline, "obs_ids")
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
def test_unified_unconditional_train_and_sample(method):
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = UnconditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1

        pipeline = UnconditionalPipeline(
            model=MockUnconditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_joint,
            method=method,
            ch_obs=2,
            training_config=training_config,
        )

        # Train 2 steps
        pipeline.train(nnx.Rngs(0), nsteps=2, save_model=False)
        pipeline._wrap_model()

        # Sample (no x_o for unconditional)
        sample = pipeline.sample(jax.random.PRNGKey(1), nsamples=32, use_ema=False)
        assert sample.shape == (32, dim_joint, 2)

        sample_ema = pipeline.sample(jax.random.PRNGKey(1), nsamples=32, use_ema=True)
        assert sample_ema.shape == (32, dim_joint, 2)

        # Batched sampling should raise
        with pytest.raises(NotImplementedError):
            pipeline.sample_batched(
                jax.random.PRNGKey(1),
                jnp.zeros((32, dim_joint, 2)),
                nsamples=20,
                chunk_size=8,
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
def test_unified_unconditional_loss_fn(method):
    pipeline = UnconditionalPipeline(
        model=MockUnconditionalModel(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_obs=dim_joint,
        method=method,
        ch_obs=2,
    )
    pipeline._wrap_model()

    loss_fn = pipeline.get_loss_fn()
    mock_batch = jnp.zeros((32, dim_joint, 2))
    loss = loss_fn(pipeline.model, mock_batch, key=jax.random.PRNGKey(1))
    assert loss.shape == ()
