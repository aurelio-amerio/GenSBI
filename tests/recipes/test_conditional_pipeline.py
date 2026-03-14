"""
Tests for ``gensbi.recipes.conditional_pipeline`` — edge cases.

Covers:
- model_extras conflict detection in get_sampler / get_log_prob_fn
- Batch x_o warning in sample()
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

import pytest
import tempfile

import grain

from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

nsamples = 200
key = jax.random.PRNGKey(0)
dim_obs = 2
dim_cond = 3

theta = jax.random.normal(key, (nsamples, dim_obs, 1))
x = jax.random.normal(key, (nsamples, dim_cond, 1))
data = jnp.concatenate([theta, x], axis=1)


def split_obs_cond(data):
    return data[:, :dim_obs], data[:, dim_obs:]


train_dataset = (
    grain.MapDataset.source(np.array(data)[:160])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)

val_dataset = (
    grain.MapDataset.source(np.array(data)[160:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
)


@pytest.fixture
def pipeline():
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir

        p = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            method=FlowMatchingMethod(),
            training_config=training_config,
        )
        p.train(nnx.Rngs(0), nsteps=2, save_model=False)
        p._wrap_model()
        yield p


# ---------------------------------------------------------------------------
# model_extras conflict detection
# ---------------------------------------------------------------------------


class TestModelExtrasConflict:
    def test_get_sampler_protected_key_raises(self, pipeline):
        """get_sampler rejects model_extras that override cond/obs_ids/cond_ids."""
        x_o = jnp.zeros((1, dim_cond, 1))
        with pytest.raises(ValueError, match="protected keys"):
            pipeline.get_sampler(
                x_o,
                model_extras={"cond": jnp.zeros((1, dim_cond, 1))},
            )

    def test_get_log_prob_fn_protected_key_raises(self, pipeline):
        """get_log_prob_fn rejects model_extras that override protected keys."""
        x_o = jnp.zeros((1, dim_cond, 1))
        with pytest.raises(ValueError, match="protected keys"):
            pipeline.get_log_prob_fn(
                x_o,
                model_extras={"obs_ids": jnp.zeros(1)},
            )


# ---------------------------------------------------------------------------
# Batch x_o warning
# ---------------------------------------------------------------------------


class TestSampleBatchWarning:
    def test_batch_xo_warns(self, pipeline):
        """sample() with batch x_o (shape > 1) emits UserWarning."""
        x_o_batch = jnp.zeros((5, dim_cond, 1))  # batch dim > 1
        with pytest.warns(UserWarning, match="batch dimension"):
            try:
                pipeline.sample(
                    jax.random.PRNGKey(1), x_o_batch, nsamples=4,
                )
            except (ValueError, Exception):
                # Mock model may fail on broadcast; we only care about the warning
                pass
