"""Sampler-kwarg handling in the unified pipelines.

Covers the GenSBI issues surfaced by the 2026-10-26 notebook re-check:

1. unknown sampler kwargs must raise (not be swallowed by ``**kwargs``),
2. ``sample_batched`` must forward the caller's ``model_extras`` to the model,
4. ``JointPipeline.sample`` must not warn on a single 1-D observation.
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import warnings
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel, MockJointModel

from gensbi.core import DiffusionEDMMethod, FlowMatchingMethod, ScoreMatchingMethod
from gensbi.recipes import ConditionalPipeline, JointPipeline

dim_obs = 2
dim_cond = 3
dim_joint = dim_obs + dim_cond

_key = jax.random.PRNGKey(0)
_data = jax.random.normal(_key, (200, dim_joint, 1))


def _split_obs_cond(d):
    return d[:, :dim_obs], d[:, dim_obs:]


def _ds_joint(arr):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(16))


def _ds_cond(arr):
    return _ds_joint(arr).map(_split_obs_cond)


def make_cond_pipeline(method=None, model=None):
    pipeline = ConditionalPipeline(
        model or MockConditionalModel(),
        _ds_cond(_data[:160]),
        _ds_cond(_data[160:]),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method or FlowMatchingMethod(),
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    return pipeline


def make_joint_pipeline():
    pipeline = JointPipeline(
        MockJointModel(),
        _ds_joint(_data[:160]),
        _ds_joint(_data[160:]),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
        condition_mask_kind="structured",
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    return pipeline


_x_o = jnp.zeros((1, dim_cond, 1))
_x_o_batch = jnp.zeros((3, dim_cond, 1))


# ---------------------------------------------------------------------------
# 1. Unknown sampler kwargs raise instead of being silently dropped
# ---------------------------------------------------------------------------


def test_fm_sample_rejects_unknown_kwarg():
    pipeline = make_cond_pipeline()
    with pytest.raises(TypeError, match="bogus_kwarg"):
        pipeline.sample(jax.random.PRNGKey(1), _x_o, nsamples=4,
                        use_ema=False, bogus_kwarg=1)


def test_edm_sample_rejects_time_grid():
    # ``time_grid`` is a flow-matching argument; the diffusion sampler must
    # not accept it silently (the two_moons diffusion notebooks relied on it).
    pipeline = make_cond_pipeline(method=DiffusionEDMMethod(sde="EDM"))
    with pytest.raises(TypeError, match="time_grid"):
        pipeline.sample(jax.random.PRNGKey(1), _x_o, nsamples=4, use_ema=False,
                        time_grid=jnp.linspace(0.0, 1.0, 5))


def test_sm_sample_rejects_time_grid():
    pipeline = make_cond_pipeline(method=ScoreMatchingMethod(sde_type="VP"))
    with pytest.raises(TypeError, match="time_grid"):
        pipeline.sample(jax.random.PRNGKey(1), _x_o, nsamples=4, use_ema=False,
                        time_grid=jnp.linspace(0.0, 1.0, 5))


def test_sample_batched_rejects_unknown_kwarg():
    # A model kwarg passed bare (not via ``model_extras``) used to be dropped
    # on the floor; the gw/lensing scripts hit this with ``encoder_key=``.
    pipeline = make_cond_pipeline()
    with pytest.raises(TypeError, match="encoder_key"):
        pipeline.sample_batched(jax.random.PRNGKey(1), _x_o_batch, nsamples=2,
                                use_ema=False, show_progress_bars=False,
                                encoder_key=jax.random.PRNGKey(0))


def test_fm_log_prob_rejects_unknown_kwarg():
    pipeline = make_cond_pipeline()
    with pytest.raises(TypeError, match="bogus_kwarg"):
        pipeline.get_log_prob_fn(_x_o, use_ema=False, bogus_kwarg=1)


# ---------------------------------------------------------------------------
# 2. sample_batched forwards the caller's model_extras to the model
# ---------------------------------------------------------------------------


class _ShiftModel(MockConditionalModel):
    """Conditional mock whose velocity is a constant ``shift`` model extra.

    Integrating a constant velocity over t in [0, 1] moves every sample by
    exactly ``shift``, so the extra is observable in the samples.
    """

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
        shift: Array | None = None,
    ) -> Array:
        out = super().__call__(t, obs, obs_ids, cond, cond_ids,
                               conditioned=conditioned, guidance=guidance)
        if shift is None:
            return out
        return out + shift


def test_sample_batched_forwards_model_extras_to_model():
    pipeline = make_cond_pipeline(model=_ShiftModel())
    key = jax.random.PRNGKey(1)
    plain = pipeline.sample_batched(key, _x_o_batch, nsamples=5, use_ema=False,
                                    show_progress_bars=False)
    shifted = pipeline.sample_batched(key, _x_o_batch, nsamples=5, use_ema=False,
                                      show_progress_bars=False,
                                      model_extras={"shift": jnp.float32(2.0)})
    assert shifted.shape == plain.shape == (5, 3, dim_obs, 1)
    # every condition in the batch must see the extra
    np.testing.assert_allclose(np.asarray(shifted - plain), 2.0, atol=1e-5)


def test_sample_batched_forwards_model_extras_when_chunked():
    pipeline = make_cond_pipeline(model=_ShiftModel())
    key = jax.random.PRNGKey(1)
    plain = pipeline.sample_batched(key, _x_o_batch, nsamples=5, use_ema=False,
                                    chunk_size=2, show_progress_bars=False)
    shifted = pipeline.sample_batched(key, _x_o_batch, nsamples=5, use_ema=False,
                                      chunk_size=2, show_progress_bars=False,
                                      model_extras={"shift": jnp.float32(2.0)})
    np.testing.assert_allclose(np.asarray(shifted - plain), 2.0, atol=1e-5)


def test_sample_batched_rejects_protected_model_extras():
    pipeline = make_cond_pipeline()
    with pytest.raises(ValueError, match="protected"):
        pipeline.sample_batched(jax.random.PRNGKey(1), _x_o_batch, nsamples=2,
                                use_ema=False, show_progress_bars=False,
                                model_extras={"cond": _x_o})


# ---------------------------------------------------------------------------
# 4. JointPipeline.sample: batch warning only for a real batch axis
# ---------------------------------------------------------------------------


def _batch_warnings(record):
    return [w for w in record if "batch dimension" in str(w.message)]


def test_joint_sample_no_batch_warning_for_1d_observation():
    pipeline = make_joint_pipeline()
    x_o = jnp.zeros((dim_cond,))  # what sbibm ``get_reference`` returns
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=4, use_ema=False)
    assert out.shape == (4, dim_obs, 1)
    assert _batch_warnings(record) == []


def test_joint_sample_no_batch_warning_for_single_3d_observation():
    pipeline = make_joint_pipeline()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        pipeline.sample(jax.random.PRNGKey(1), _x_o, nsamples=4, use_ema=False)
    assert _batch_warnings(record) == []


def test_joint_sample_warns_for_batched_observations():
    pipeline = make_joint_pipeline()
    with pytest.warns(UserWarning, match="batch dimension 2"):
        pipeline.sample(jax.random.PRNGKey(1), jnp.zeros((2, dim_cond, 1)),
                        nsamples=2, use_ema=False)
