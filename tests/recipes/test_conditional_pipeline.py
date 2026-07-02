"""
Tests for ``gensbi.recipes.conditional_pipeline`` — edge cases.

Covers:
- model_extras conflict detection in get_sampler / get_log_prob_fn
- Single-observation policy: batched x_o raises; 1-D x_o is promoted
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


class TestSingleObservationPolicy:
    def test_sample_batch_xo_raises(self, pipeline):
        """Batched x_o raises: single-observation methods never silently
        discard observations (reverses the 4cc400b warn+take-first policy)."""
        x_o_batch = jnp.zeros((5, dim_cond, 1))
        with pytest.raises(ValueError, match="single observation"):
            pipeline.sample(jax.random.PRNGKey(1), x_o_batch, nsamples=4)

    def test_get_sampler_batch_xo_raises(self, pipeline):
        with pytest.raises(ValueError, match="single observation"):
            pipeline.get_sampler(jnp.zeros((5, dim_cond, 1)))

    def test_get_log_prob_fn_batch_xo_raises(self, pipeline):
        with pytest.raises(ValueError, match="single observation"):
            pipeline.get_log_prob_fn(jnp.zeros((5, dim_cond, 1)))

    def test_sample_1d_xo_promoted_not_truncated(self, pipeline):
        """Regression (review Finding 2): a bare (dim_cond,) observation is
        promoted to (1, dim_cond, 1) — not read as a batch and truncated to
        its first scalar coordinate."""
        s = pipeline.sample(jax.random.PRNGKey(1), jnp.zeros(dim_cond), nsamples=4)
        assert s.shape[0] == 4

    def test_sample_batched_unaffected(self, pipeline, recwarn):
        x_o = jnp.zeros((3, dim_cond, 1))
        pipeline.sample_batched(jax.random.PRNGKey(2), x_o, 4,
                                show_progress_bars=False)
        assert not any("batch" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# Patch-size threading tests
# ---------------------------------------------------------------------------


def test_conditional_pipeline_patch_size():
    """size threads to obs_ids for a 2D obs strategy; cond (1D) is unaffected."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir

        p = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=(16, 16),
            dim_cond=3,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("rope2d", "absolute"),
            size=8,
            training_config=training_config,
        )
        # obs: 16//8 * 16//8 = 4 patch tokens
        assert p.obs_ids.shape[1] == 4
        assert p.dim_obs == 4
        # cond is 1D -> size ignored, 3 tokens
        assert p.cond_ids.shape[1] == 3
        assert p.dim_cond == 3


def test_conditional_pipeline_patch_size_tuple():
    """A tuple lets obs and cond differ; cond 1D still ignores its entry."""
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = ConditionalPipeline.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir

        p = ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=(16, 16),
            dim_cond=3,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("rope2d", "absolute"),
            size=(8, 1),
            training_config=training_config,
        )
        assert p.obs_ids.shape[1] == 4
        assert p.cond_ids.shape[1] == 3
