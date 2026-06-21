import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.experimental.recipes import FieldConditionalWrapper


class _EchoModel(nnx.Module):
    """Records the shapes it was called with and returns obs unchanged."""

    def __init__(self):
        self.seen = {}

    def __call__(self, *, t, obs, cond, obs_ids=None, cond_ids=None,
                 conditioned=True, guidance=None):
        self.seen = dict(t=t.shape, obs=obs.shape, cond=cond.shape)
        return obs


def test_wrapper_batches_unbatched_field_and_cond():
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    out = w(t=jnp.array(0.5), obs=jnp.ones((8, 8, 1)), cond=jnp.ones((3,)))
    assert m.seen["obs"] == (1, 8, 8, 1)   # (H,W,C) -> (1,H,W,C)
    assert m.seen["cond"] == (1, 3)        # (k,) -> (1,k)
    assert out.shape == (1, 8, 8, 1)


def test_wrapper_passes_batched_inputs_through():
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    w(t=jnp.ones((4,)), obs=jnp.ones((4, 8, 8, 1)), cond=jnp.ones((4, 3, 1)))
    assert m.seen["obs"] == (4, 8, 8, 1)
    assert m.seen["cond"] == (4, 3, 1)


def test_wrapper_broadcasts_batch1_cond_to_obs_batch():
    """Sampling N draws for one x_o: obs arrives batch-N, cond batch-1."""
    m = _EchoModel()
    w = FieldConditionalWrapper(m)
    w(t=jnp.ones((4,)), obs=jnp.ones((4, 8, 8, 1)), cond=jnp.ones((1, 3, 1)))
    assert m.seen["cond"] == (4, 3, 1)


import tempfile

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.models import FieldDiT, FieldDiTParams
from gensbi.experimental.recipes import FieldConditionalPipeline

H = W = 16
COND_DIM = 3


class _Loop:
    """Iterable dataset yielding the same (obs, cond) batch forever."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def _tiny_fielddit(seed=0):
    return FieldDiT(FieldDiTParams(
        in_channels=1,
        field_shape=(H, W),
        encoder_widths=(4, 8),       # D = 1
        cond_dim=COND_DIM,
        rngs=nnx.Rngs(seed),
        res_blocks_down=1,
        res_blocks_up=1,
        patch_size=2,
        num_heads=2,
        axes_dim=[2, 2, 4],          # hidden 16
        depth=1,
        depth_single_blocks=1,
        param_dtype=jnp.float32,
    ))


def _make_pipeline(model_dir, seed=0):
    key = jax.random.PRNGKey(seed)
    obs = jax.random.normal(key, (32, H, W, 1))
    cond = jax.random.normal(jax.random.fold_in(key, 1), (32, COND_DIM, 1))
    batch = (obs, cond)

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config["checkpoint_dir"] = model_dir

    return FieldConditionalPipeline(
        model=_tiny_fielddit(seed),
        train_dataset=_Loop(batch),
        val_dataset=_Loop(batch),
        field_shape=(H, W),
        dim_cond=COND_DIM,
        method=FlowMatchingMethod(),
        ch_obs=1,
        training_config=training_config,
    )


def test_pipeline_constructs_with_field_event_shape():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        assert p.event_shape == (H, W, 1)
        assert p.method.prior.event_shape == (H, W, 1)
        assert p.obs_ids is None and p.cond_ids is None


def test_pipeline_trains_two_steps():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        before = jax.tree_util.tree_leaves(nnx.state(p.model, nnx.Param))
        before = [leaf.copy() for leaf in before]
        p.train(nnx.Rngs(0), nsteps=2, save_model=False)
        after = jax.tree_util.tree_leaves(nnx.state(p.model, nnx.Param))
        changed = any(
            not jnp.array_equal(b, a) for b, a in zip(before, after)
        )
        assert changed


def test_pipeline_samples_field_shaped_output():
    with tempfile.TemporaryDirectory() as model_dir:
        p = _make_pipeline(model_dir)
        p.train(nnx.Rngs(0), nsteps=2, save_model=False)
        p._wrap_model()
        x_o = jnp.ones((1, COND_DIM, 1))
        samples = p.sample(jax.random.PRNGKey(0), x_o, nsamples=4, step_size=0.5)
        assert samples.shape == (4, H, W, 1)
        assert jnp.all(jnp.isfinite(samples))
