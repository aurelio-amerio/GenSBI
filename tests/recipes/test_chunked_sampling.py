"""Tests for nsamples-chunked sampling (spec 2026-07-22-chunked-sampling-design)."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import grain
import pytest

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel, MockJointModel, MockUnconditionalModel

from gensbi.recipes import (
    ConditionalPipeline,
    JointPipeline,
    UnconditionalPipeline,
)
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod
from gensbi.recipes.pipeline import _chunked_draw, _sample_concat_axis


# ---------------------------------------------------------------------------
# Unit tests: _chunked_draw with a spy sampler
# ---------------------------------------------------------------------------


class _SpySampler:
    """Records every call; returns constant arrays of the requested size."""

    def __init__(self, extra_leading=None):
        self.calls = []          # list of (key, n, kwargs)
        self.extra_leading = extra_leading

    def __call__(self, key, n, **kwargs):
        self.calls.append((key, n, kwargs))
        if self.extra_leading is None:
            return jnp.full((n, 3, 1), float(len(self.calls)))
        return jnp.full((self.extra_leading, n, 3, 1), float(len(self.calls)))


def test_chunked_draw_none_is_single_call_with_original_key():
    spy = _SpySampler()
    key = jax.random.PRNGKey(0)
    out = _chunked_draw(spy, key, 100, None, show_progress_bars=False)
    assert out.shape == (100, 3, 1)
    assert len(spy.calls) == 1
    called_key, n, _ = spy.calls[0]
    assert n == 100
    assert jnp.array_equal(called_key, key)  # bit-identical path: original key


def test_chunked_draw_large_chunk_is_single_call_with_original_key():
    spy = _SpySampler()
    key = jax.random.PRNGKey(1)
    out = _chunked_draw(spy, key, 10, 100, show_progress_bars=False)
    assert out.shape == (10, 3, 1)
    assert len(spy.calls) == 1
    assert jnp.array_equal(spy.calls[0][0], key)


def test_chunked_draw_remainder_chunks():
    spy = _SpySampler()
    out = _chunked_draw(spy, jax.random.PRNGKey(2), 25, 10,
                        show_progress_bars=False)
    assert out.shape == (25, 3, 1)
    assert [c[1] for c in spy.calls] == [10, 10, 5]
    # each chunk got a distinct key
    keys = [tuple(np.asarray(c[0]).tolist()) for c in spy.calls]
    assert len(set(keys)) == 3
    # chunks were concatenated in call order along axis 0
    assert jnp.all(out[:10] == 1.0)
    assert jnp.all(out[10:20] == 2.0)
    assert jnp.all(out[20:] == 3.0)


def test_chunked_draw_concat_axis_1_for_intermediates():
    spy = _SpySampler(extra_leading=4)  # (n_steps=4, n, 3, 1)
    out = _chunked_draw(spy, jax.random.PRNGKey(3), 25, 10,
                        show_progress_bars=False, concat_axis=1)
    assert out.shape == (4, 25, 3, 1)


def test_chunked_draw_forwards_sampler_kwargs():
    spy = _SpySampler()
    extras = {"model_extras": {"cond": jnp.zeros((1, 2, 1))}}
    _chunked_draw(spy, jax.random.PRNGKey(4), 25, 10,
                  show_progress_bars=False, sampler_kwargs=extras)
    assert all(c[2] == extras for c in spy.calls)


def test_chunked_draw_external_pbar_updated_per_chunk():
    class _FakeBar:
        def __init__(self):
            self.n = 0

        def update(self, k):
            self.n += k

    spy = _SpySampler()
    bar = _FakeBar()
    _chunked_draw(spy, jax.random.PRNGKey(5), 25, 10,
                  show_progress_bars=False, pbar=bar)
    assert bar.n == 3
    bar2 = _FakeBar()
    _chunked_draw(spy, jax.random.PRNGKey(6), 25, None,
                  show_progress_bars=False, pbar=bar2)
    assert bar2.n == 1  # single-call path still ticks an external bar once


def test_sample_concat_axis():
    assert _sample_concat_axis({}) == 0
    assert _sample_concat_axis({"return_intermediates": True}) == 1
    assert _sample_concat_axis({"return_intermediates": False}) == 0
    # FlowMatchingMethod: any non-None time_grid turns intermediates on
    assert _sample_concat_axis({"time_grid": jnp.linspace(0, 1, 5)}) == 1
    assert _sample_concat_axis({"time_grid": None}) == 0


def test_get_batch_sampler_removed():
    with pytest.raises(ImportError):
        from gensbi.recipes.pipeline import _get_batch_sampler  # noqa: F401


# ---------------------------------------------------------------------------
# Shared pipeline fixtures (used by pipeline-level tests, Tasks 2-4)
# ---------------------------------------------------------------------------

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond

_key = jax.random.PRNGKey(0)
_theta = jax.random.normal(_key, (200, dim_obs, 2))
_x = jax.random.normal(_key, (200, dim_cond, 2))
_data = jnp.concatenate([_theta, _x], axis=1)


def _split_obs_cond(d):
    return d[:, :dim_obs], d[:, dim_obs:]


def _ds_joint(arr):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(16))


def _ds_cond(arr):
    return _ds_joint(arr).map(_split_obs_cond)


def make_cond_pipeline(method=None):
    pipeline = ConditionalPipeline(
        MockConditionalModel(),
        _ds_cond(_data[:160]),
        _ds_cond(_data[160:]),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=method or FlowMatchingMethod(),
        ch_obs=2,
        ch_cond=2,
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    return pipeline
