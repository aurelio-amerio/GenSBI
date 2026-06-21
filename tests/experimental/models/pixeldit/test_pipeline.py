"""Pipeline integration tests for PixelDiT: construct / train-step / sample.

Uses FieldConditionalPipeline with a tiny PixelDiT config and an in-memory
looping dataset (obs (B,16,16,1), cond (B,2,1)).  No production code is
changed; PixelDiT already ignores obs_ids/cond_ids passed via extras.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import tempfile

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.models import PixelDiT, PixelDiTParams
from gensbi.experimental.recipes import FieldConditionalPipeline

H = W = 16
COND_DIM = 2
BATCH = 8


class _Loop:
    """Iterable dataset yielding the same (obs, cond) batch forever."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def _tiny_pixeldit(seed=0):
    return PixelDiT(PixelDiTParams(
        in_channels=1,
        field_shape=(H, W),
        cond_dim=COND_DIM,
        rngs=nnx.Rngs(seed),
        hidden_size=64,
        num_heads=4,
        patch_depth=2,
        pixel_depth=2,
        patch_size=4,
        pixel_hidden_size=8,
        param_dtype=jnp.float32,
    ))


def _make_pipeline(model_dir, seed=0):
    key = jax.random.PRNGKey(seed)
    obs = jax.random.normal(key, (BATCH, H, W, 1))
    cond = jax.random.normal(jax.random.fold_in(key, 1), (BATCH, COND_DIM, 1))
    batch = (obs, cond)

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config["checkpoint_dir"] = model_dir

    return FieldConditionalPipeline(
        model=_tiny_pixeldit(seed),
        train_dataset=_Loop(batch),
        val_dataset=_Loop(batch),
        field_shape=(H, W),
        dim_cond=COND_DIM,
        method=FlowMatchingMethod(),
        ch_obs=1,
        training_config=training_config,
    )


# ---------------------------------------------------------------------------
# Test 1: construct + loss
# ---------------------------------------------------------------------------


def test_pipeline_constructs_and_loss_is_finite():
    """Pipeline constructs and get_loss_fn() returns a finite scalar loss."""
    with tempfile.TemporaryDirectory() as model_dir:
        pipeline = _make_pipeline(model_dir)

        assert pipeline.event_shape == (H, W, 1)
        assert pipeline.obs_ids is None
        assert pipeline.cond_ids is None

        pipeline._wrap_model()
        loss_fn = pipeline.get_loss_fn()

        key = jax.random.PRNGKey(42)
        obs = jax.random.normal(key, (BATCH, H, W, 1))
        cond = jax.random.normal(jax.random.fold_in(key, 1), (BATCH, COND_DIM, 1))
        batch = (obs, cond)

        loss = loss_fn(pipeline.model_wrapped, batch, key=jax.random.PRNGKey(1))
        assert loss.shape == ()
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 2: three optimizer steps, loss finite throughout
# ---------------------------------------------------------------------------


def test_three_optimizer_steps_change_weights():
    """3 training steps run, loss is finite throughout, and weights are updated."""
    with tempfile.TemporaryDirectory() as model_dir:
        pipeline = _make_pipeline(model_dir)
        # val_every=1 ensures losses are recorded on every step so we can
        # assert finiteness; default val_every=100 would yield an empty list
        # for only 3 steps.
        pipeline.training_config["val_every"] = 1
        before = jax.tree_util.tree_leaves(nnx.state(pipeline.model, nnx.Param))
        before = [leaf.copy() for leaf in before]
        loss_array, _val_loss_array = pipeline.train(nnx.Rngs(0), nsteps=3, save_model=False)
        after = jax.tree_util.tree_leaves(nnx.state(pipeline.model, nnx.Param))
        assert len(loss_array) > 0, "no train losses recorded"
        assert all(jnp.isfinite(l) for l in loss_array), "non-finite train loss"
        changed = any(not jnp.array_equal(b, a) for b, a in zip(before, after))
        assert changed


# ---------------------------------------------------------------------------
# Test 3: sample returns correct shape and finite values
# ---------------------------------------------------------------------------


def test_sample_shape_and_finite():
    """pipeline.sample returns (nsamples, H, W, 1) finite array."""
    with tempfile.TemporaryDirectory() as model_dir:
        pipeline = _make_pipeline(model_dir)
        pipeline._wrap_model()

        cond = jnp.ones((1, COND_DIM, 1))
        samples = pipeline.sample(
            jax.random.PRNGKey(0), cond, nsamples=2, step_size=0.5
        )
        assert samples.shape == (2, H, W, 1)
        assert jnp.all(jnp.isfinite(samples))
