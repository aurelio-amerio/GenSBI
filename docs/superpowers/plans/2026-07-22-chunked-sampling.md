# Chunked Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-bounded chunking over the `nsamples` dimension to `sample()` and `sample_batched()` across all pipelines, replacing the dead `chunk_size` parameter and the unused `_get_batch_sampler` helper.

**Architecture:** One shared helper `_chunked_draw` in `src/gensbi/recipes/pipeline.py` wraps the existing `(key, nsamples) -> samples` sampler closures. Every `sample()` delegates its final draw to it; `AbstractPipeline.sample_batched` keeps its outer per-condition loop and calls the helper per condition; `ConditionalFlowPipeline.sample_batched` chunks its flattened `B*nsamples` autoregressive batch directly. Spec: `docs/superpowers/specs/2026-07-22-chunked-sampling-design.md`.

**Tech Stack:** JAX, flax.nnx, tqdm, pytest (CPU: test files set `JAX_PLATFORMS=cpu`).

## Global Constraints

- **Backward compatibility is bit-exact:** `chunk_size=None` (the new default everywhere) or `chunk_size >= nsamples` must call the sampler ONCE with the ORIGINAL key — identical to current behavior. Never split the key on that path.
- `chunk_size` means "maximum samples per device call" in every pipeline. Default `None` = no chunking.
- Intermediates detection: EDM/SM methods use an explicit `return_intermediates=True` kwarg; FlowMatchingMethod turns intermediates ON whenever `time_grid is not None` (see `src/gensbi/core/flow_matching.py:253-257`). Both must map to `concat_axis=1`.
- Run tests with the mamba `gensbi` environment (NOT `.venv`): `mamba run -n gensbi python -m pytest ...` (if `mamba run` is unavailable, use `conda run -n gensbi`). Working dir: `/lhome/ific/a/aamerio/data/github/GenSBI`.
- Keep `nsamples` small (≤ 30) in tests — everything runs on CPU.
- Commit after every task. Branch: create `chunked-sampling` off `main` before Task 1 (`git checkout -b chunked-sampling`). Do NOT touch the unrelated dirty file `src/gensbi/models/flux1/model.py`.

---

### Task 1: `_chunked_draw` + `_sample_concat_axis` helpers; delete `_get_batch_sampler`

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py:136-201` (replace `_get_batch_sampler` with the two new helpers)
- Modify: `tests/recipes/test_pipeline_edge_cases.py:16` (drop stale import) and `tests/recipes/test_pipeline_edge_cases.py:376-391` (delete obsolete test)
- Create: `tests/recipes/test_chunked_sampling.py`

**Interfaces:**
- Produces: `_chunked_draw(sampler, key, nsamples, chunk_size, show_progress_bars=True, concat_axis=0, sampler_kwargs=None, pbar=None) -> Array` and `_sample_concat_axis(sampler_kwargs: dict) -> int`, both importable as `from gensbi.recipes.pipeline import _chunked_draw, _sample_concat_axis`. All later tasks consume these.
- Consumes: nothing new. `sampler` is any closure with signature `sampler(key, nsamples, **kwargs) -> Array`.

- [ ] **Step 1: Write the failing tests**

Create `tests/recipes/test_chunked_sampling.py` with this exact content (the pipeline-level fixtures at the bottom are used by Tasks 2–4; include them now so the file is complete):

```python
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
```

- [ ] **Step 2: Run the unit tests to verify they fail**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -v
```

Expected: FAIL — `ImportError: cannot import name '_chunked_draw'`.

- [ ] **Step 3: Implement the helpers**

In `src/gensbi/recipes/pipeline.py`, DELETE the whole `_get_batch_sampler` function (lines 136–201, from `def _get_batch_sampler(` through `    return sampler` inclusive) and put this in its place:

```python
def _sample_concat_axis(sampler_kwargs: dict) -> int:
    """Axis carrying the sample dimension in a sampler's output.

    Solvers stack intermediates along a leading, statically-sized time
    axis, so chunked outputs must concatenate along axis 1 instead of 0.
    Intermediates are requested either explicitly
    (``return_intermediates=True`` — EDM and score-matching methods) or
    implicitly by passing a non-``None`` ``time_grid``
    (``FlowMatchingMethod.build_sampler_fn`` turns intermediates on for
    any explicit time grid).
    """
    if sampler_kwargs.get("return_intermediates", False):
        return 1
    if sampler_kwargs.get("time_grid", None) is not None:
        return 1
    return 0


def _chunked_draw(
    sampler: Callable,
    key: Array,
    nsamples: int,
    chunk_size: Optional[int],
    show_progress_bars: bool = True,
    concat_axis: int = 0,
    sampler_kwargs: Optional[dict] = None,
    pbar=None,
):
    """Draw ``nsamples`` from ``sampler`` in memory-bounded chunks.

    Parameters
    ----------
    sampler : Callable
        ``sampler(key, nsamples, **sampler_kwargs) -> Array``.
    key : jax.random.PRNGKey
        Random key. With no chunking it is passed through UNCHANGED so
        the result is bit-identical to calling ``sampler`` directly.
    nsamples : int
        Total number of samples to draw.
    chunk_size : int or None
        Maximum samples per sampler call. ``None`` (or any value
        ``>= nsamples``) disables chunking.
    show_progress_bars : bool, optional
        Show a tqdm bar over chunks (only when chunking is active and no
        external ``pbar`` is supplied).
    concat_axis : int, optional
        Axis to concatenate chunks along — 0 for plain samples, 1 when
        the sampler returns intermediates with a leading time axis (see
        :func:`_sample_concat_axis`).
    sampler_kwargs : dict, optional
        Extra keyword arguments forwarded to every sampler call
        (e.g. ``{"model_extras": ...}``).
    pbar : tqdm-like, optional
        External progress bar; when given it is updated once per chunk
        and no internal bar is created (used by ``sample_batched`` for a
        single bar across conditions).

    Returns
    -------
    Array
        ``nsamples`` samples, concatenated along ``concat_axis``.
    """
    kwargs = sampler_kwargs or {}

    if chunk_size is None or chunk_size >= nsamples:
        out = sampler(key, nsamples, **kwargs)
        if pbar is not None:
            out = jax.block_until_ready(out)
            pbar.update(1)
        return out

    n_chunks = (nsamples + chunk_size - 1) // chunk_size
    keys = jax.random.split(key, n_chunks)

    own_bar = pbar is None and show_progress_bars
    if own_bar:
        pbar = tqdm(total=n_chunks, desc="Sampling")

    results = []
    remaining = nsamples
    for i in range(n_chunks):
        n_i = min(chunk_size, remaining)
        remaining -= n_i
        chunk = sampler(keys[i], n_i, **kwargs)
        # Wait for the device so the progress bar is accurate and host
        # memory for the next chunk isn't requested while this one runs.
        chunk = jax.block_until_ready(chunk)
        results.append(chunk)
        if pbar is not None:
            pbar.update(1)

    if own_bar:
        pbar.close()

    return jnp.concatenate(results, axis=concat_axis)
```

Then in `tests/recipes/test_pipeline_edge_cases.py`:
- Delete line 16: `from gensbi.recipes.pipeline import _get_batch_sampler`
- Delete the whole function `test_get_batch_sampler_no_progress_bars` (lines 376–391, from `def test_get_batch_sampler_no_progress_bars():` through `    assert result.shape == (n_samples, ncond, 1)` inclusive).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py tests/recipes/test_pipeline_edge_cases.py -v
```

Expected: all PASS (edge-cases file no longer references the deleted helper).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/pipeline.py tests/recipes/test_chunked_sampling.py tests/recipes/test_pipeline_edge_cases.py
git commit -m "feat: add _chunked_draw/_sample_concat_axis, drop unused _get_batch_sampler"
```

---

### Task 2: Rewire `AbstractPipeline.sample_batched`

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py:905-963` (the `sample_batched` method)
- Test: `tests/recipes/test_chunked_sampling.py`

**Interfaces:**
- Consumes: `_chunked_draw`, `_sample_concat_axis` (Task 1); `make_cond_pipeline` fixture (Task 1's test file).
- Produces: `AbstractPipeline.sample_batched(self, key, x_o, nsamples, *args, chunk_size=None, show_progress_bars=True, **kwargs)` — same output contract as before: `(nsamples, B, dim_obs, ch_obs)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recipes/test_chunked_sampling.py`:

```python
# ---------------------------------------------------------------------------
# AbstractPipeline.sample_batched: per-condition loop + nsamples chunking
# ---------------------------------------------------------------------------


def test_sample_batched_chunked_shape_and_default_equivalence():
    pipeline = make_cond_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (3, dim_cond, 2))
    key = jax.random.PRNGKey(1)

    ref = pipeline.sample_batched(key, x_o, nsamples=10, use_ema=False,
                                  show_progress_bars=False)
    assert ref.shape == (10, 3, dim_obs, 2)

    chunked = pipeline.sample_batched(key, x_o, nsamples=10, use_ema=False,
                                      chunk_size=4, show_progress_bars=False)
    assert chunked.shape == (10, 3, dim_obs, 2)
    assert jnp.all(jnp.isfinite(chunked))

    # chunk_size >= nsamples short-circuits to the unchunked path:
    # bit-identical to chunk_size=None with the same key
    big = pipeline.sample_batched(key, x_o, nsamples=10, use_ema=False,
                                  chunk_size=64, show_progress_bars=False)
    assert jnp.array_equal(big, ref)


def test_sample_batched_chunk_size_default_is_none():
    import inspect
    sig = inspect.signature(AbstractPipeline.sample_batched)
    assert sig.parameters["chunk_size"].default is None


def test_sample_batched_progress_bar_smoke():
    # progress-bar branch must not crash (tqdm writes to stderr)
    pipeline = make_cond_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (2, dim_cond, 2))
    out = pipeline.sample_batched(jax.random.PRNGKey(1), x_o, nsamples=6,
                                  use_ema=False, chunk_size=4,
                                  show_progress_bars=True)
    assert out.shape == (6, 2, dim_obs, 2)
```

Add `AbstractPipeline` to the existing import in the test file:

```python
from gensbi.recipes.pipeline import _chunked_draw, _sample_concat_axis, AbstractPipeline
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -k sample_batched -v
```

Expected: FAIL — `test_sample_batched_chunk_size_default_is_none` fails (`default is 50`), the equivalence test fails only if the implementation errors (it currently ignores `chunk_size`, so it may pass — the signature test is the true red here).

- [ ] **Step 3: Replace the `sample_batched` body**

In `src/gensbi/recipes/pipeline.py`, replace the whole `sample_batched` method (lines 905–963) with:

```python
    def sample_batched(
        self,
        key,
        x_o: Array,
        nsamples: int,
        *args,
        chunk_size: Optional[int] = None,
        show_progress_bars=True,
        **kwargs,
    ):
        """
        Generate samples from the trained model in batches.

        Loops over the ``B`` conditions in ``x_o`` one at a time and, when
        ``chunk_size`` is set, additionally draws each condition's samples
        in memory-bounded chunks of at most ``chunk_size`` samples per
        device call.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable (e.g., observed data), leading batch
            axis of size ``B``.
        nsamples : int
            Number of samples to generate per condition.
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws all ``nsamples`` for a condition in a single
            call — identical to the historical behavior.
        show_progress_bars : bool, optional
            Whether to display a progress bar over the
            ``B * n_chunks`` device calls. Default is True.
        args : tuple
            Additional positional arguments for the sampler.
        kwargs : dict
            Additional keyword arguments for the sampler.

        Returns
        -------
        samples : array-like
            Generated samples of shape (nsamples, batch_size_cond, dim_obs, ch_obs).
        """

        # Build the sampler once using the first condition for shape.
        # The sampler's JIT compilation traces model_extras by shape/dtype,
        # so calling it with different cond values (same shape) reuses the
        # compiled function — no recompilation per condition.
        sampler = self.get_sampler(x_o[0:1], *args, **kwargs)

        B = x_o.shape[0]
        keys_per_cond = jax.random.split(key, B)

        if chunk_size is None or chunk_size >= nsamples:
            n_chunks_per_cond = 1
        else:
            n_chunks_per_cond = (nsamples + chunk_size - 1) // chunk_size
        pbar = (
            tqdm(total=B * n_chunks_per_cond, desc="Sampling")
            if show_progress_bars
            else None
        )
        concat_axis = _sample_concat_axis(kwargs)

        results = []
        for i in range(B):
            cond_i = _expand_dims(x_o[i : i + 1])
            extras_i = {
                "cond": cond_i,
                "obs_ids": self.obs_ids,
                "cond_ids": self.cond_ids,
            }
            samples_i = _chunked_draw(
                sampler,
                keys_per_cond[i],
                nsamples,
                chunk_size,
                concat_axis=concat_axis,
                sampler_kwargs={"model_extras": extras_i},
                pbar=pbar,
            )
            results.append(samples_i)
        if pbar is not None:
            pbar.close()

        return jnp.stack(results, axis=1)  # (nsamples, B, dim_obs, ch_obs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -v
mamba run -n gensbi python -m pytest tests/recipes/test_solver_edm_pipelines.py tests/recipes/test_pipeline_edge_cases.py -v
```

Expected: all PASS (the EDM sample_batched tests exercise the `model_extras` swap path and must be unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/pipeline.py tests/recipes/test_chunked_sampling.py
git commit -m "feat: nsamples chunking in AbstractPipeline.sample_batched (chunk_size finally live)"
```

---

### Task 3: Chunking in `ConditionalPipeline.sample` and `UnconditionalPipeline.sample`

**Files:**
- Modify: `src/gensbi/recipes/conditional_pipeline.py:258-281` (`sample`)
- Modify: `src/gensbi/recipes/unconditional_pipeline.py:190-210` (`sample`)
- Test: `tests/recipes/test_chunked_sampling.py`

**Interfaces:**
- Consumes: `_chunked_draw`, `_sample_concat_axis` from `gensbi.recipes.pipeline` (Task 1).
- Produces: `ConditionalPipeline.sample(self, key, x_o, nsamples=10_000, use_ema=True, chunk_size=None, show_progress_bars=True, **sampler_kwargs)`; `UnconditionalPipeline.sample(self, key, nsamples=10_000, use_ema=True, chunk_size=None, show_progress_bars=True, **sampler_kwargs)`. Return contracts unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recipes/test_chunked_sampling.py`:

```python
# ---------------------------------------------------------------------------
# sample(): conditional + unconditional
# ---------------------------------------------------------------------------


def test_conditional_sample_chunked_shape():
    pipeline = make_cond_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=10,
                          use_ema=False, chunk_size=4,
                          show_progress_bars=False)
    assert out.shape == (10, dim_obs, 2)
    assert jnp.all(jnp.isfinite(out))


def test_conditional_sample_chunked_bit_identical_when_not_chunking():
    pipeline = make_cond_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    key = jax.random.PRNGKey(1)
    ref = pipeline.sample(key, x_o, nsamples=10, use_ema=False)
    big = pipeline.sample(key, x_o, nsamples=10, use_ema=False,
                          chunk_size=999)
    assert jnp.array_equal(big, ref)


def test_conditional_sample_chunked_with_edm_intermediates():
    # EDM + return_intermediates: output (n_steps, nsamples, dim, ch);
    # chunks must concatenate along the SAMPLE axis (1), not the time axis.
    pipeline = make_cond_pipeline(method=DiffusionEDMMethod(sde="EDM"))
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    ref = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=9,
                          use_ema=False, nsteps=4, return_intermediates=True)
    out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=9,
                          use_ema=False, nsteps=4, return_intermediates=True,
                          chunk_size=4, show_progress_bars=False)
    assert out.shape == ref.shape          # same (n_steps, 9, dim_obs, 2)
    assert out.shape[1] == 9               # sample axis grew to nsamples
    assert out.ndim == 4


def test_conditional_sample_chunked_with_fm_time_grid_intermediates():
    # FlowMatchingMethod: a non-None time_grid implicitly enables
    # intermediates (core/flow_matching.py:253-257).
    pipeline = make_cond_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    tg = jnp.linspace(0.0, 1.0, 5)
    ref = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=9,
                          use_ema=False, time_grid=tg)
    out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=9,
                          use_ema=False, time_grid=tg,
                          chunk_size=4, show_progress_bars=False)
    assert out.shape == ref.shape
    assert out.shape[1] == 9


def test_unconditional_sample_chunked_shape():
    pipeline = UnconditionalPipeline(
        model=MockUnconditionalModel(),
        train_dataset=_ds_joint(_data[:160]),
        val_dataset=_ds_joint(_data[160:]),
        dim_obs=dim_joint,
        method=FlowMatchingMethod(),
        ch_obs=2,
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    out = pipeline.sample(jax.random.PRNGKey(1), nsamples=10, use_ema=False,
                          chunk_size=4, show_progress_bars=False)
    assert out.shape == (10, dim_joint, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -k "conditional_sample or unconditional_sample" -v
```

Expected: FAIL — `TypeError: sample() got an unexpected keyword argument 'chunk_size'` (it currently lands in `**sampler_kwargs` and reaches `build_sampler_fn`, which rejects it — either way, red).

- [ ] **Step 3: Implement**

In `src/gensbi/recipes/conditional_pipeline.py`, add to the imports near the top (it already imports from `gensbi.recipes.pipeline`; extend that import or add a line):

```python
from gensbi.recipes.pipeline import _chunked_draw, _sample_concat_axis
```

Replace the `sample` method (lines 258–281) with:

```python
    def sample(self, key, x_o, nsamples=10_000, use_ema=True,
               chunk_size=None, show_progress_bars=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : array-like
            Conditioning variable.
        nsamples : int, optional
            Number of samples. Default is 10 000.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws everything in one call — identical to the
            historical behavior. Set it to bound memory when drawing
            many samples from a large model.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """

        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return _chunked_draw(
            sampler, key, nsamples, chunk_size,
            show_progress_bars=show_progress_bars,
            concat_axis=_sample_concat_axis(sampler_kwargs),
        )
```

In `src/gensbi/recipes/unconditional_pipeline.py`, extend the existing `from gensbi.recipes.pipeline import AbstractPipeline` (line 31) to:

```python
from gensbi.recipes.pipeline import AbstractPipeline, _chunked_draw, _sample_concat_axis
```

Replace the `sample` method (lines 190–210) with:

```python
    def sample(self, key, nsamples=10_000, use_ema=True,
               chunk_size=None, show_progress_bars=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        nsamples : int, optional
            Number of samples. Default is 10 000.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws everything in one call — identical to the
            historical behavior.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """
        sampler = self.get_sampler(use_ema=use_ema, **sampler_kwargs)
        return _chunked_draw(
            sampler, key, nsamples, chunk_size,
            show_progress_bars=show_progress_bars,
            concat_axis=_sample_concat_axis(sampler_kwargs),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -v
mamba run -n gensbi python -m pytest tests/recipes/test_unified_conditional_pipeline.py tests/recipes/test_unified_unconditional_pipeline.py tests/recipes/test_solver_fm_pipelines.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/conditional_pipeline.py src/gensbi/recipes/unconditional_pipeline.py tests/recipes/test_chunked_sampling.py
git commit -m "feat: chunk_size in ConditionalPipeline.sample and UnconditionalPipeline.sample"
```

---

### Task 4: Chunking in `JointPipeline.sample` + Simformer passthroughs

**Files:**
- Modify: `src/gensbi/recipes/joint_pipeline.py:390-422` (`sample`)
- Modify: `src/gensbi/recipes/simformer.py` — the three `sample` overrides (`SimformerFlowPipeline.sample` ~line 262, `SimformerSMPipeline.sample` ~line 403, `SimformerDiffusionPipeline.sample` ~line 563)
- Test: `tests/recipes/test_chunked_sampling.py`

**Interfaces:**
- Consumes: `_chunked_draw`, `_sample_concat_axis` (Task 1).
- Produces: `JointPipeline.sample(self, key, x_o, nsamples=10_000, use_ema=True, chunk_size=None, show_progress_bars=True, **sampler_kwargs)`; the three Simformer `sample` overrides gain explicit `chunk_size=None, show_progress_bars=True` parameters forwarded to `super().sample`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recipes/test_chunked_sampling.py`:

```python
# ---------------------------------------------------------------------------
# sample(): joint + simformer passthroughs
# ---------------------------------------------------------------------------


def make_joint_pipeline():
    pipeline = JointPipeline(
        MockJointModel(),
        _ds_joint(_data[:160]),
        _ds_joint(_data[160:]),
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        method=FlowMatchingMethod(),
        ch_obs=2,
        condition_mask_kind="structured",
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    return pipeline


def test_joint_sample_chunked_shape():
    pipeline = make_joint_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=10,
                          use_ema=False, chunk_size=4,
                          show_progress_bars=False)
    assert out.shape == (10, dim_obs, 2)


def test_joint_sample_chunked_bit_identical_when_not_chunking():
    pipeline = make_joint_pipeline()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    key = jax.random.PRNGKey(1)
    ref = pipeline.sample(key, x_o, nsamples=10, use_ema=False)
    big = pipeline.sample(key, x_o, nsamples=10, use_ema=False,
                          chunk_size=999)
    assert jnp.array_equal(big, ref)


def test_simformer_sample_signatures_accept_chunk_kwargs():
    import inspect
    from gensbi.recipes import SimformerFlowPipeline, SimformerDiffusionPipeline
    from gensbi.recipes.simformer import SimformerSMPipeline

    for cls in (SimformerFlowPipeline, SimformerSMPipeline,
                SimformerDiffusionPipeline):
        sig = inspect.signature(cls.sample)
        assert "chunk_size" in sig.parameters, cls.__name__
        assert sig.parameters["chunk_size"].default is None, cls.__name__
        assert "show_progress_bars" in sig.parameters, cls.__name__


def test_simformer_flow_sample_chunked_end_to_end():
    from flax import nnx
    from gensbi.models import SimformerParams
    from gensbi.recipes import SimformerFlowPipeline

    params = SimformerParams(
        rngs=nnx.Rngs(0), in_channels=2, val_emb_dim=2, id_emb_dim=2,
        cond_emb_dim=2, dim_joint=dim_joint, fourier_features=32,
        num_heads=2, depth=1, mlp_ratio=1, qkv_features=4,
        num_hidden_layers=1,
    )
    pipeline = SimformerFlowPipeline(
        _ds_joint(_data[:160]), _ds_joint(_data[160:]),
        dim_obs, dim_cond, ch_obs=2, params=params,
    )
    pipeline.ema_model = pipeline.model
    pipeline._wrap_model()
    x_o = jax.random.normal(jax.random.PRNGKey(2), (1, dim_cond, 2))
    out = pipeline.sample(jax.random.PRNGKey(1), x_o, nsamples=6,
                          use_ema=False, chunk_size=4,
                          show_progress_bars=False)
    assert out.shape == (6, dim_obs, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -k "joint or simformer" -v
```

Expected: FAIL — joint tests with `TypeError` on `chunk_size`; simformer signature test with `AssertionError`.

- [ ] **Step 3: Implement**

In `src/gensbi/recipes/joint_pipeline.py`, extend the pipeline import (the file already has `from gensbi.recipes.pipeline import AbstractPipeline` — check the exact line and extend it):

```python
from gensbi.recipes.pipeline import AbstractPipeline, _chunked_draw, _sample_concat_axis
```

Replace the `sample` method (lines 390–422) with (keep the existing multi-condition warning verbatim):

```python
    def sample(self, key, x_o, nsamples=10_000, use_ema=True,
               chunk_size=None, show_progress_bars=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : array-like
            Conditioning variable.
        nsamples : int, optional
            Number of samples. Default is 10 000.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws everything in one call — identical to the
            historical behavior.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """

        x_o_shape = x_o.shape[0] if hasattr(x_o, "shape") else len(x_o)
        if x_o_shape > 1:
            warnings.warn(
                f"x_o has batch dimension {x_o_shape} > 1. "
                "sample() draws all samples for a single condition. "
                "To sample for multiple conditions, use sample_batched() instead.",
                UserWarning,
                stacklevel=2,
            )
        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return _chunked_draw(
            sampler, key, nsamples, chunk_size,
            show_progress_bars=show_progress_bars,
            concat_axis=_sample_concat_axis(sampler_kwargs),
        )
```

In `src/gensbi/recipes/simformer.py`, update the three overrides (house style: explicit params, no `**kwargs`).

`SimformerFlowPipeline.sample` (~line 262):

```python
    def sample(
        self, key, x_o, nsamples=10_000, step_size=0.01, use_ema=True,
        time_grid=None, chunk_size=None, show_progress_bars=True,
    ):
        return super().sample(
            key,
            x_o,
            nsamples=nsamples,
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
            model_extras={"edge_mask": self.edge_mask},
        )
```

`SimformerSMPipeline.sample` (~line 403):

```python
    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        nsteps=1000,
        use_ema=True,
        return_intermediates=False,
        chunk_size=None,
        show_progress_bars=True,
    ):
        return super().sample(
            key,
            x_o,
            nsamples=nsamples,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
            model_extras={"edge_mask": self.edge_mask},
        )
```

`SimformerDiffusionPipeline.sample` (~line 563):

```python
    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        chunk_size=None,
        show_progress_bars=True,
    ):
        return super().sample(
            key,
            x_o,
            nsamples=nsamples,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
            model_extras={"edge_mask": self.edge_mask},
        )
```

(Before editing, confirm the current default is `nsteps=18` at that line — keep every existing default exactly as found; only the two new parameters are additions.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
mamba run -n gensbi python -m pytest tests/recipes/test_chunked_sampling.py -v
mamba run -n gensbi python -m pytest tests/recipes/test_unified_joint_pipeline.py tests/recipes/test_pipeline_simformer.py tests/recipes/test_solver_sm_pipelines.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/joint_pipeline.py src/gensbi/recipes/simformer.py tests/recipes/test_chunked_sampling.py
git commit -m "feat: chunk_size in JointPipeline.sample + Simformer passthroughs"
```

---

### Task 5: Chunking in `ConditionalFlowPipeline` (MAF/TarFlow)

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py:247-330` (`sample` and `sample_batched`) and its imports (lines 8–13)
- Test: `tests/normalizing_flows/test_flow_pipeline.py`

**Interfaces:**
- Consumes: `_chunked_draw` (Task 1). The flow pipeline has no solver, hence no intermediates — `concat_axis` is always 0 here.
- Produces: `ConditionalFlowPipeline.sample(self, key, x_o, nsamples=10_000, use_ema=True, chunk_size=None, show_progress_bars=True, **kwargs)`; `ConditionalFlowPipeline.sample_batched(self, key, x_o, nsamples=10_000, *, use_ema=True, chunk_size=None, show_progress_bars=True, **kwargs)` chunking the flattened `B*nsamples` batch. Output shapes unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/normalizing_flows/test_flow_pipeline.py` (reuses that file's `build_pipeline`, `_EchoFlow`, `DIM_OBS`, `DIM_COND` fixtures):

```python
def test_sample_chunked_shape():
    pipe = build_pipeline()
    x_o = jnp.zeros((1, DIM_COND, 1))
    out = pipe.sample(jax.random.PRNGKey(0), x_o, nsamples=10,
                      use_ema=False, chunk_size=4, show_progress_bars=False)
    assert out.shape == (10, DIM_OBS, 1)
    assert jnp.all(jnp.isfinite(out))


def test_sample_chunked_bit_identical_when_not_chunking():
    pipe = build_pipeline()
    x_o = jnp.zeros((1, DIM_COND, 1))
    key = jax.random.PRNGKey(0)
    ref = pipe.sample(key, x_o, nsamples=10, use_ema=False)
    big = pipe.sample(key, x_o, nsamples=10, use_ema=False, chunk_size=999)
    assert jnp.array_equal(big, ref)


def test_sample_batched_chunked_shape_with_real_flow():
    pipe = build_pipeline()
    out = pipe.sample_batched(jax.random.PRNGKey(0),
                              jnp.zeros((2, DIM_COND, 1)), 7,
                              chunk_size=5, show_progress_bars=False)
    assert out.shape == (7, 2, DIM_OBS, 1)
    assert jnp.all(jnp.isfinite(out))


def test_sample_batched_chunked_routing_across_chunk_boundaries():
    # chunk_size=7 does NOT divide nsamples=5 or B*nsamples=15: chunk
    # boundaries fall inside conditions. Routing condition i -> column i
    # must survive the flattened chunking.
    pipe = build_pipeline()
    pipe.ema_model = _EchoFlow()
    B, nsamples = 3, 5
    x_o = jnp.stack([jnp.full((DIM_COND, 1), float(i)) for i in range(B)])
    out = pipe.sample_batched(jax.random.PRNGKey(0), x_o, nsamples,
                              chunk_size=7, show_progress_bars=False)
    assert out.shape == (nsamples, B, DIM_COND, 1)
    for i in range(B):
        assert jnp.all(out[:, i] == float(i))


def test_sample_batched_chunked_bit_identical_when_not_chunking():
    pipe = build_pipeline()
    x_o = jnp.zeros((2, DIM_COND, 1))
    key = jax.random.PRNGKey(0)
    ref = pipe.sample_batched(key, x_o, nsamples=7)
    big = pipe.sample_batched(key, x_o, nsamples=7, chunk_size=10_000)
    assert jnp.array_equal(big, ref)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mamba run -n gensbi python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k chunked -v
```

Expected: FAIL. Note the failure mode: `chunk_size`/`show_progress_bars` currently fall into `**kwargs` and trigger the `_warn_unused_kwargs` UserWarning while being ignored — assertions on shape may pass but `test_sample_batched_chunked_routing_across_chunk_boundaries` and the bit-identical tests still red-flag via warning-free behavior change. If any of these tests accidentally passes before implementing, tighten it rather than skipping the red step: the routing test is the load-bearing one.

- [ ] **Step 3: Implement**

In `src/gensbi/recipes/flow_pipeline.py`, update the imports (currently lines 8–13) to add `jax` and `tqdm` and the helper:

```python
import warnings

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from gensbi.recipes.pipeline import AbstractPipeline, _chunked_draw
from gensbi.recipes.utils import _require_channel, _single_obs
```

Replace `sample` (lines 247–275) with:

```python
    def sample(self, key, x_o, nsamples=10_000, use_ema=True,
               chunk_size=None, show_progress_bars=True, **kwargs):
        """Draw posterior samples for a single conditioning observation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : Array
            Single conditioning observation carrying a leading batch axis of
            size 1 (see :meth:`get_sampler` for the shape convention). A leading
            batch axis > 1 raises ``ValueError``.
        nsamples : int, optional
            Number of posterior samples to draw.  Default is 10 000.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws everything in one call — identical to the
            historical behavior. Set it to bound memory when drawing many
            samples from a deep flow.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(nsamples, dim_obs, 1)`` for the
            tabular default (``C = 1``), or ``(nsamples, dim_obs, C)`` for
            ``ch_obs = C`` — the channel axis is always carried for a
            vector-modeled variable regardless of ``structured_cond`` (a
            structured condition changes only ``x_o``'s expected shape, not
            the modeled variable's). When ``structured_obs=True``, samples
            instead have shape ``(nsamples,) + per_obs_shape``, the model's
            native structured output.
        """
        sampler = self.get_sampler(x_o, use_ema=use_ema, **kwargs)
        return _chunked_draw(
            sampler, key, nsamples, chunk_size,
            show_progress_bars=show_progress_bars,
        )
```

Replace `sample_batched` (lines 277–330) with:

```python
    def sample_batched(self, key, x_o, nsamples=10_000, *, use_ema=True,
                       chunk_size=None, show_progress_bars=True, **kwargs):
        """Draw posterior samples for a batch of conditioning observations.

        Each condition is repeated ``nsamples`` times and concatenated
        into a single flattened ``(B * nsamples, ...)`` batch. Without
        ``chunk_size`` the whole batch runs in **one** autoregressive
        pass (memory scales with ``B * nsamples``); with ``chunk_size``
        the flattened batch is sliced into pieces of at most
        ``chunk_size`` rows per ``flow.sample`` call — chunk boundaries
        may fall inside a condition, which is fine because every row is
        independent.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for the batched sampling pass.
        x_o : Array
            Batch of observations. For tabular cond: shape
            ``(B, dim_cond, C)`` (a bare ``(B, dim_cond)`` raises
            ``ValueError`` — add a trailing channel axis). For structured
            cond: ``(B,) + per_obs_shape``.
        nsamples : int, optional
            Number of posterior samples per observation.  Default is
            10 000.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.
        chunk_size : int, optional
            Maximum number of rows of the flattened ``B * nsamples``
            batch per device call. ``None`` (default) keeps the
            historical single-pass behavior.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.
        **kwargs : dict, optional
            Extra keyword arguments accepted for interface compatibility
            and ignored with a warning (e.g. solver arguments from
            :class:`~gensbi.recipes.pipeline.AbstractPipeline`).

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(nsamples, B, dim_obs, 1)`` for the
            tabular default (``C = 1``), or ``(nsamples, B, dim_obs, C)`` for
            ``ch_obs = C``. When ``structured_obs=True``, samples instead
            have shape ``(nsamples, B) + per_obs_shape``. In both cases
            ``out[:, i]`` is the samples for condition ``i``.
        """
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        x_o = jnp.asarray(x_o)
        if not self.structured_cond:
            x_o = _require_channel(x_o, "x_o")
        B = x_o.shape[0]
        cond = jnp.repeat(x_o, nsamples, axis=0)  # (B*nsamples, ...): c0 x nsamples, c1 x nsamples, ...
        total = B * nsamples

        if chunk_size is None or chunk_size >= total:
            samples = flow.sample(key, cond=cond)  # ONE batched AR pass
        else:
            n_chunks = (total + chunk_size - 1) // chunk_size
            keys = jax.random.split(key, n_chunks)
            loop = range(n_chunks)
            if show_progress_bars:
                loop = tqdm(loop, desc="Sampling")
            chunks = []
            for i in loop:
                sl = slice(i * chunk_size, min((i + 1) * chunk_size, total))
                chunk = flow.sample(keys[i], cond=cond[sl])
                chunks.append(jax.block_until_ready(chunk))
            samples = jnp.concatenate(chunks, axis=0)

        samples = samples.reshape((B, nsamples) + samples.shape[1:])
        return jnp.moveaxis(samples, 0, 1)  # (nsamples, B, dim_obs, C)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
mamba run -n gensbi python -m pytest tests/normalizing_flows/test_flow_pipeline.py -v
mamba run -n gensbi python -m pytest tests/normalizing_flows/ tests/models/maf/ tests/models/tarflow/ -q
```

Expected: all PASS (MAF/TarFlow suites confirm no regression in the flows the pipeline wraps).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat: chunked sampling in ConditionalFlowPipeline (flattened B*nsamples chunks)"
```

---

### Task 6: Full-suite verification

**Files:**
- No source changes expected; fix regressions if any surface.

**Interfaces:**
- Consumes: everything above.
- Produces: green suite; branch ready for review/merge.

- [ ] **Step 1: Run the full recipes + flows test surface**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI
mamba run -n gensbi python -m pytest tests/recipes/ tests/normalizing_flows/ tests/models/ -q
```

Expected: all PASS. If anything fails, fix it (it is a regression from Tasks 1–5 — do not skip or xfail) and re-run.

- [ ] **Step 2: Spec cross-check**

Re-read `docs/superpowers/specs/2026-07-22-chunked-sampling-design.md` section by section and confirm each is implemented: helper semantics (§`_chunked_draw`), `sample()` surface (§`sample()` gains chunking — conditional, joint, flow, unconditional, 3 Simformer overrides), `AbstractPipeline.sample_batched` (chunk_size live, default None, single bar over `B × n_chunks`), `FlowPipeline.sample_batched` (flattened chunking), `chunk_size` semantics (max samples per device call, None default). Fix any gap found.

- [ ] **Step 3: Final commit (if Step 1/2 required changes)**

```bash
git add -A -- src/ tests/
git commit -m "fix: chunked-sampling suite regressions"
```

Only commit if there were changes; otherwise the branch already ends on Task 5's commit.
