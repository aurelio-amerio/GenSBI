# maf-branch Merge Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all merge-blocking fixes from the 2026-07-02 code review (spec: `docs/superpowers/specs/2026-07-02-maf-merge-fixes-design.md`) so the `maf` branch is mergeable once the owner's GPU recovery gates pass.

**Architecture:** Nine sequential work units on the `maf` branch, correctness first: (1) the `_rescale` MCLMC formula, (2) a shared `_single_obs` observation-canonicalization helper with a hard-error batch policy replacing warn+take-first, (3) recovery-script repair, then extras (batched `sample_batched`, MetaBlock `inv_perm` removal, `fit_stat` hoist), hygiene (deprecating re-export, gitlink removal), and a docs pass anchored on a new `docs/advanced/normalizing_flows.md` page.

**Tech Stack:** JAX / flax nnx, blackjax (MCLMC), pytest, Sphinx/MyST docs.

## Global Constraints

- Repo root: `/lustre/ific.uv.es/ml/ific088/github/GenSBI`, branch `maf`. All paths below are repo-relative.
- Run all tests in the **mamba `gensbi` env**, never `.venv`: `mamba run -n gensbi python -m pytest <args>`. (The two envs differ; only `gensbi` surfaces real failures.)
- Tests default to CPU (`JAX_PLATFORMS=cpu` is set in `pyproject.toml` pytest config; `addopts = "-n 2"`).
- Fast test gate for each task: `mamba run -n gensbi python -m pytest -m "not slow"` must be green before committing.
- One commit per task, message prefixes as given. End commit messages with the session trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013AH3qcKsN7zQ2jKaxDaVNY`.
- The **final merge gate is owned by the user**: full GPU recovery runs of all four `scripts/*_recovery.py` after Task 1 lands. Do not merge to `main` in this plan.

---

### Task 1: Fix `_rescale` (blackjax adjusted-MCLMC step-count formula)

**Files:**
- Modify: `src/gensbi/inference/samplers.py:10-38` (`_rescale`, `_check_rescale_domain`)
- Test: `tests/inference/test_mclmc.py`

**Interfaces:**
- Produces: `_rescale(mu) -> float array` such that `E[ceil(U(0,1) * _rescale(mu))] == mu` exactly, for `mu >= 1`. `_check_rescale_domain(mu)` raises `ValueError` for `mu < 1`.
- Consumers (unchanged): `MCLMC._run_adjusted` calls `_rescale` at `samplers.py:221` (tuning kernel) and `:235` (final sampler), and `_check_rescale_domain` at `:230`.

**Background:** adjusted MCLMC randomizes the number of integration steps per proposal as `ceil(U(0,1) * s)` to avoid trajectory resonances. Given the tuned mean `mu = L/step_size`, `_rescale` must return the `s` with `E[ceil(U*s)] = mu`. Current code returns `mu / round(log2(2*mu - 1))` — a transcription bug (at `mu=15` it yields `s=3`, so proposals average ~2 steps instead of ~15, silently degrading every default `NLEPosterior.sample()` call).

- [ ] **Step 1: Write the failing tests**

In `tests/inference/test_mclmc.py`, add `import math` at the top, add `_rescale` to the existing `from gensbi.inference.samplers import _check_rescale_domain` import, and add:

```python
def _expected_ceil_uniform(s):
    """Closed-form E[ceil(U(0,1) * s)] for s = k + frac, k = floor(s)."""
    k = math.floor(s)
    frac = s - k
    return (k * (k + 1) / 2 + frac * (k + 1)) / s


@pytest.mark.parametrize("mu", [1.0, 1.5, 5.3, 15.0])
def test_rescale_gives_exact_mean_step_count(mu):
    s = float(_rescale(mu))
    assert _expected_ceil_uniform(s) == pytest.approx(mu, rel=1e-6)


def test_rescale_matches_blackjax_reference_at_15():
    # blackjax adjusted_mclmc_dynamic: k = floor(2mu-1); x = k(mu-(k+1)/2)/(k+1-mu).
    # At mu=15 the fractional part is 0, so s = 2*mu - 1 = 29 exactly.
    assert float(_rescale(15.0)) == pytest.approx(29.0)
```

Also REPLACE the existing `test_check_rescale_domain_guard` (which asserts `mu=0.5+1e-3` is accepted — the old `log`-based domain) with:

```python
def test_check_rescale_domain_guard():
    _check_rescale_domain(2.0)      # fine
    _check_rescale_domain(1.0)      # boundary: s=1, always one step
    for bad in (0.999, 0.5, 0.0, -3.0):
        with pytest.raises(ValueError, match="L/step_size"):
            _check_rescale_domain(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: the new `test_rescale_*` tests FAIL (wrong values), `test_check_rescale_domain_guard` FAILS (0.999 does not raise).

- [ ] **Step 3: Implement**

In `src/gensbi/inference/samplers.py`, replace `_rescale` and `_check_rescale_domain` entirely with:

```python
def _rescale(mu):
    """Map a mean trajectory length to a uniform-integer draw scale.

    From blackjax's ``adjusted_mclmc_dynamic``: drawing the number of
    integration steps as ``ceil(U(0,1) * _rescale(L/step_size))`` makes the
    average number of steps exactly ``mu = L / step_size``.

    ``mu`` must satisfy ``mu >= 1``; see :func:`_check_rescale_domain` for the
    host-side guard applied to the tuned value.
    """
    k = jnp.floor(2 * mu - 1)
    x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
    return k + x


def _check_rescale_domain(mu):
    """Raise if the tuned ``mu = L / step_size`` is outside ``_rescale``'s domain.

    For ``mu < 1``, ``floor(2 * mu - 1) == 0`` and ``_rescale`` returns 0, so
    the integration-step draw ``ceil(U(0,1) * 0)`` is 0 — a chain that never
    moves. A host-side check on the tuned value turns that silent failure into
    an explicit error. (The in-tuning average is left to blackjax; this is a
    convenience sampler, not a fully hardened MCMC engine.)
    """
    mu = float(mu)
    if mu < 1.0:
        raise ValueError(
            f"adjusted-MCLMC tuning produced L/step_size = {mu:.4g} < 1, for "
            f"which the randomized integration-step count rounds to zero and "
            f"the chain would never move. This usually means tuning did not "
            f"converge — try increasing num_tuning_steps, increasing "
            f"num_samples, or using MCLMC(adjusted=False).")
```

Note: the old `jax.lax.max` usage disappears; check whether `jax` (bare) is still used elsewhere in the module before touching imports (it is — leave imports alone).

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/inference/ -v`
Expected: ALL PASS (including the existing MCLMC smoke tests, which now run correct-length trajectories).

- [ ] **Step 5: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/inference/samplers.py tests/inference/test_mclmc.py
git commit -m "fix(inference): correct _rescale to blackjax adjusted-MCLMC step-count formula"
```

---
do rd-error batch policy

**Files:**
- Modify: `src/gensbi/recipes/utils.py` (add `_require_channel`, `_single_obs`)
- Modify: `src/gensbi/recipes/flow_pipeline.py` (delete local `_require_channel`/`_warn_if_batched`/`_single_obs`; import from utils; update 2 call sites + docstrings)
- Modify: `src/gensbi/recipes/conditional_pipeline.py` (delete `_single_cond_fm`; update 2 call sites; drop dead imports)
- Test: `tests/normalizing_flows/test_flow_pipeline.py`, `tests/recipes/test_conditional_pipeline.py`

**Interfaces:**
- Produces: `_single_obs(x_o, *, channel, name="x_o") -> Array` in `gensbi/recipes/utils.py`. Canonicalizes shape FIRST (`channel` mode: `"require"` = enforce `(B, dim, C)` via `_require_channel`; `"promote"` = lenient `_expand_dims` promotion `(dim,) -> (1, dim, 1)`, `(B, dim) -> (B, dim, 1)`; `"none"` = structured, only `ndim >= 2` required), THEN raises `ValueError` if the leading batch axis is > 1. Returns the canonicalized array **with its size-1 batch axis kept**.
- Produces: `_require_channel(x, name="input")` moves verbatim from `flow_pipeline.py` to `recipes/utils.py` (flow_pipeline re-imports it, so `from gensbi.recipes.flow_pipeline import _require_channel` keeps working).
- Behavior change: batched `x_o` (B>1) in `get_sampler`/`get_log_prob_fn`/`sample`/`log_prob` of BOTH pipelines now raises `ValueError` instead of warn+take-first (reverses commit `4cc400b`; owner-approved).

- [ ] **Step 1: Write the failing tests — flow pipeline**

In `tests/normalizing_flows/test_flow_pipeline.py`:

Change the import at line ~12-14 to:

```python
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.recipes.utils import _require_channel, _single_obs
```

REPLACE the three existing `_single_obs` unit tests (`test_single_obs_keeps_channel_strips_batch`, `test_single_obs_batched_warns_and_takes_first`, `test_single_obs_rejects_rank_lt_2`) with:

```python
def test_single_obs_require_keeps_batch_and_channel():
    out = _single_obs(jnp.zeros((1, DIM_COND, 1)), channel="require")
    assert out.shape == (1, DIM_COND, 1)


def test_single_obs_none_keeps_structured_shape():
    img = jnp.arange(1 * 1 * 4 * 2).reshape(1, 1, 4, 2)
    assert _single_obs(img, channel="none").shape == (1, 1, 4, 2)


def test_single_obs_batched_raises():
    x_o = jnp.arange(3 * DIM_COND).reshape(3, DIM_COND, 1)
    with pytest.raises(ValueError, match="single observation"):
        _single_obs(x_o, channel="require")
    with pytest.raises(ValueError, match="single observation"):
        _single_obs(x_o, channel="none")


def test_single_obs_require_rejects_channelless():
    # (1, dim): documented contract violation -> the class-docstring ValueError
    with pytest.raises(ValueError, match="channel axis"):
        _single_obs(jnp.zeros((1, DIM_COND)), channel="require")
    # (dim, C): must NOT be misread as `dim` observations (review Finding 3)
    with pytest.raises(ValueError, match="channel axis"):
        _single_obs(jnp.zeros((DIM_COND, 2)), channel="require")


def test_single_obs_promote_1d_and_2d():
    assert _single_obs(jnp.zeros((DIM_COND,)), channel="promote").shape == (1, DIM_COND, 1)
    assert _single_obs(jnp.zeros((1, DIM_COND)), channel="promote").shape == (1, DIM_COND, 1)


def test_single_obs_none_rejects_rank_lt_2():
    with pytest.raises(ValueError):
        _single_obs(jnp.zeros((DIM_COND,)), channel="none")
```

ADD pipeline-level tests (the error now fires in `_single_obs` before any model call, so one MAFlow-backed pipeline covers TarFlow too):

```python
def test_get_sampler_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_sampler(jnp.zeros((1, DIM_COND)))
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_sampler(jnp.zeros((DIM_COND, 2)))


def test_get_log_prob_fn_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.get_log_prob_fn(jnp.zeros((1, DIM_COND)))


def test_get_sampler_batched_xo_raises():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="single observation"):
        pipe.get_sampler(jnp.zeros((5, DIM_COND, 1)))
```

If the file has other tests relying on warn+take-first (search for `pytest.warns(UserWarning, match="batch dimension")`), convert them to `pytest.raises(ValueError, match="single observation")`.

- [ ] **Step 2: Write the failing tests — FM pipeline**

In `tests/recipes/test_conditional_pipeline.py`, REPLACE the `TestSampleBatchWarning` class with:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `mamba run -n gensbi python -m pytest tests/normalizing_flows/test_flow_pipeline.py tests/recipes/test_conditional_pipeline.py -v`
Expected: new/changed tests FAIL (`_single_obs` has no `channel` kwarg; batched inputs warn instead of raising; 1-D FM input crashes downstream).

- [ ] **Step 4: Implement the shared helper**

In `src/gensbi/recipes/utils.py`, add near the top (after the existing imports):

```python
import warnings

from gensbi.utils.math import _expand_dims
```

and add (place after `init_ids_2d`, before `scale_lr`):

```python
def _require_channel(x, name="input"):
    """Enforce a tabular channel axis (B, dim, C); reject a bare (B, dim)."""
    x = jnp.asarray(x)
    if x.ndim < 3:
        raise ValueError(
            f"{name} must carry a channel axis (B, dim, C); got shape "
            f"{tuple(x.shape)}. A bare (B, dim) is not accepted — add a trailing "
            f"channel axis (e.g. x[..., None] for C=1).")
    return x


def _single_obs(x_o, *, channel, name="x_o"):
    """Canonicalize a single conditioning observation, then enforce batch == 1.

    Shape handling comes FIRST so a misshaped input can never be misread as a
    batch (e.g. ``(dim, C)`` read as ``dim`` observations):

    - ``channel="require"``: tabular flow-pipeline contract — input must
      already carry batch and channel axes ``(1, dim, C)``; channel-less input
      raises ``ValueError`` (same :func:`_require_channel` as training).
    - ``channel="promote"``: FM-pipeline contract — lenient promotion:
      ``(dim,) -> (1, dim, 1)`` and ``(B, dim) -> (B, dim, 1)``.
    - ``channel="none"``: structured inputs — the model owns the trailing
      shape; only a leading batch axis (``ndim >= 2``) is required.

    A leading batch axis > 1 then raises ``ValueError``: single-observation
    methods never silently discard observations — use ``sample_batched``.
    Returns the canonicalized array with its size-1 batch axis kept.
    """
    x_o = jnp.asarray(x_o)
    if channel == "require":
        x_o = _require_channel(x_o, name)
    elif channel == "promote":
        x_o = _expand_dims(x_o)
        if x_o.ndim < 3:
            raise ValueError(
                f"{name} must be at least 1-D (dim,); got shape {tuple(x_o.shape)}.")
    elif channel == "none":
        if x_o.ndim < 2:
            raise ValueError(
                f"{name} must carry a leading batch axis (e.g. (1,) + "
                f"per_observation_shape); got shape {tuple(x_o.shape)}.")
    else:
        raise ValueError(f"unknown channel mode {channel!r}")
    if x_o.shape[0] > 1:
        raise ValueError(
            f"{name} has a leading batch axis of size {x_o.shape[0]} > 1, but "
            "this method conditions on a single observation and will not "
            "silently discard the rest. Use sample_batched() for a batch of "
            "conditions.")
    return x_o
```

- [ ] **Step 5: Rewire `flow_pipeline.py`**

- Delete the local `_require_channel` (lines 16-24), `_warn_if_batched` (27-35), and `_single_obs` (38-47) definitions.
- Add to imports: `from gensbi.recipes.utils import _require_channel, _single_obs` (keep the `warnings` import — `_warn_unused_kwargs` and `train` still use it).
- In `get_sampler` (line ~270) replace `cond = _single_obs(x_o)` with:

```python
        mode = "none" if self.structured_cond else "require"
        cond = _single_obs(x_o, channel=mode)[0]  # (cond_dim, C_cond) or structured per-obs shape
```

- In `get_log_prob_fn` (line ~374) replace `cond = _single_obs(x_o)` with the same two lines.
- Docstring updates (warn semantics are gone):
  - Class `Notes` (line ~96-97): replace "A batch axis > 1 emits a ``UserWarning`` and the first observation is used — pass a batch to :meth:`sample_batched` instead." with "A batch axis > 1 raises ``ValueError`` — pass a batch to :meth:`sample_batched` instead."
  - In `get_sampler`, `sample`, `get_log_prob_fn`, `log_prob` parameter docs: replace every "A leading batch axis > 1 emits a ``UserWarning`` and the first observation is used" / "A batch axis > 1 warns and the first observation is used" sentence with "A leading batch axis > 1 raises ``ValueError``" (keep the pointer to :meth:`sample_batched`).

- [ ] **Step 6: Rewire `conditional_pipeline.py`**

- Delete `_single_cond_fm` (lines 52-71).
- Add `_single_obs` to the existing `from gensbi.recipes.utils import (...)` block.
- In `get_sampler` (line ~245): replace `cond = _expand_dims(_single_cond_fm(x_o))` with `cond = _single_obs(x_o, channel="promote")`.
- In `get_log_prob_fn` (line ~321): same replacement.
- Remove now-dead imports if unused elsewhere in the file: `_expand_dims` (from `gensbi.utils.model_wrapping`) and `import warnings` — verify with a grep in the file before removing each.

- [ ] **Step 7: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/normalizing_flows tests/recipes tests/models/tarflow tests/diagnostics -v`
Expected: ALL PASS. If any other test still expects the old warning (`match="batch dimension"`), convert it to `pytest.raises(ValueError, match="single observation")` — each conversion is this same policy change, not a new decision.

- [ ] **Step 8: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/recipes/utils.py src/gensbi/recipes/flow_pipeline.py src/gensbi/recipes/conditional_pipeline.py tests/normalizing_flows/test_flow_pipeline.py tests/recipes/test_conditional_pipeline.py
git commit -m "fix(recipes): canonicalize x_o before batch check; hard error on batched single-obs input"
```

---

### Task 3: Repair `tarflow_image_npe_recovery.py`

**Files:**
- Modify: `scripts/tarflow_image_npe_recovery.py:73` and `:93`

**Interfaces:**
- Consumes: `TarFlowParams(cond=...)` valid values are `"bias"`/`"vector"`/`"image"` (post-rename); `pipe.sample()` returns `(nsamples, D, 1)`.

- [ ] **Step 1: Fix the two lines**

- Line ~73: `cond="image_prefix",` → `cond="image",`
- Line ~93: `s = pipe.sample(jax.random.PRNGKey(7), x_o, nsamples=num_samples)` → `s = pipe.sample(jax.random.PRNGKey(7), x_o, nsamples=num_samples)[..., 0]` (the channel squeeze its three sibling scripts apply; `jnp.cov(s.T)` needs 2-D input and `mean_s` must be `(D,)` to compare with the analytic mean).

- [ ] **Step 2: Verify by executing (the review confirmed the bug this way)**

Run: `mamba run -n gensbi python scripts/tarflow_image_npe_recovery.py --smoke`
Expected: trains a few steps on CPU and prints `SMOKE OK` (exit 0). Before the fix it dies at construction with `ValueError: unknown cond 'image_prefix'`.

- [ ] **Step 3: Commit**

```bash
git add scripts/tarflow_image_npe_recovery.py
git commit -m "fix(scripts): repair tarflow_image_npe_recovery (cond rename + channel squeeze)"
```

---

### Task 4: Batched `sample_batched` (one AR pass instead of B)

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py:301-348` (`sample_batched`)
- Test: `tests/normalizing_flows/test_flow_pipeline.py`

**Interfaces:**
- Produces: `sample_batched(key, x_o, nsamples=10_000, *, use_ema=True, **kwargs) -> (nsamples, B, dim_obs, C)`; `x_o` must be `(B, dim_cond, C)` for tabular (`_require_channel` enforced) or `(B,) + per_obs_shape` for structured. Output layout identical to the old per-condition loop: `out[:, i]` are the samples for condition `i`.
- Consumes: `MAFlow.sample`/`TarFlow.sample` accept a batched `cond` of shape `(N, dim_cond, C)` and return `(N, dim_obs, C)` (already true — `get_sampler` relies on it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/normalizing_flows/test_flow_pipeline.py`:

```python
class _EchoFlow:
    """Stub flow whose sample() echoes its condition — makes the
    condition->column routing of sample_batched exactly checkable."""
    def sample(self, key, cond):
        return cond


def test_sample_batched_routes_each_condition_to_its_column():
    pipe = build_pipeline()
    pipe.ema_model = _EchoFlow()
    B, nsamples = 3, 5
    x_o = jnp.stack([jnp.full((DIM_COND, 1), float(i)) for i in range(B)])
    out = pipe.sample_batched(jax.random.PRNGKey(0), x_o, nsamples)
    assert out.shape == (nsamples, B, DIM_COND, 1)
    for i in range(B):
        assert jnp.all(out[:, i] == float(i))


def test_sample_batched_shape_with_real_flow():
    pipe = build_pipeline()
    out = pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND, 1)), 7)
    assert out.shape == (7, 2, DIM_OBS, 1)


def test_sample_batched_rejects_channelless_xo():
    pipe = build_pipeline()
    with pytest.raises(ValueError, match="channel axis"):
        pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND)), 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mamba run -n gensbi python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k sample_batched -v`
Expected: echo test FAILS (current loop calls `get_sampler` which broadcasts, works — but channel-less test FAILS because old code sliced without validation; run and record actual failures; the echo test also fails because `_EchoFlow.sample` is called with `cond=` shape `(nsamples, DIM_COND, 1)` per column — output shape mismatch on stack? If the echo test happens to pass under the old loop, that is fine: it pins behavior the rewrite must preserve.)

- [ ] **Step 3: Implement**

Replace the body of `sample_batched` (keep the method signature) with:

```python
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        x_o = jnp.asarray(x_o)
        if not self.structured_cond:
            x_o = _require_channel(x_o, "x_o")
        B = x_o.shape[0]
        cond = jnp.repeat(x_o, nsamples, axis=0)      # (B*nsamples, ...): c0 x nsamples, c1 x nsamples, ...
        samples = flow.sample(key, cond=cond)          # (B*nsamples, dim_obs, C) — ONE batched AR pass
        samples = samples.reshape((B, nsamples) + samples.shape[1:])
        return jnp.moveaxis(samples, 0, 1)             # (nsamples, B, dim_obs, C)
```

Update the docstring: delete the "Loops the single-observation sampler..." paragraph; state that all `B` conditions are drawn in **one** batched autoregressive pass (`B * nsamples` rows — memory scales with the product); `x_o` must be `(B, dim_cond, C)` for tabular cond (a bare `(B, dim_cond)` raises `ValueError`) or `(B,) + per_obs_shape` for structured; return shape/layout unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/normalizing_flows tests/diagnostics -v`
Expected: ALL PASS (diagnostics exercise `sample_batched` through SBC/TARP).

- [ ] **Step 5: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "perf(recipes): single batched AR pass in ConditionalFlowPipeline.sample_batched"
```

---

### Task 5: MetaBlock derives `inv_perm` internally

**Files:**
- Modify: `src/gensbi/models/tarflow/blocks.py:128-136` (constructor + docstring)
- Modify: `src/gensbi/models/tarflow/model.py:191-192` (call site)
- Test: `tests/models/tarflow/test_blocks_meta.py`, `tests/models/tarflow/test_stability.py`

**Interfaces:**
- Produces: `MetaBlock(F, channels, T, perm, conditioner, num_layers, num_heads, expansion, rngs, zero_init=True, use_softplus=True, soft_clip=4.0)` — `inv_perm` parameter REMOVED; computed internally as `argsort(perm)` (as the sibling `Permutation` bijection already does). `self.inv_perm` attribute unchanged (still a `Mask`), so forward/inverse code is untouched.

- [ ] **Step 1: Update tests to the new signature (the failing step)**

- `tests/models/tarflow/test_blocks_meta.py`: the module builder at line ~13-15 computes `inv_perm = jnp.argsort(perm)` and passes `inv_perm=inv_perm`; a second construction at line ~86 passes `inv_perm=jnp.argsort(perm)`. Remove the `inv_perm=` argument (and the now-unused local) from both.
- `tests/models/tarflow/test_stability.py:18`: remove `inv_perm=jnp.arange(4),`.
- Add to `test_blocks_meta.py` (reusing the module's existing builder — adapt the call to its actual name/signature):

```python
def test_metablock_derives_inverse_permutation():
    perm = jax.random.permutation(jax.random.PRNGKey(3), 4)
    block = _make_block(perm=perm)  # the module's existing MetaBlock builder
    assert jnp.array_equal(block.inv_perm[...], jnp.argsort(perm))
```

If the builder does not take a `perm` argument, add one (default `jnp.arange(T)`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `mamba run -n gensbi python -m pytest tests/models/tarflow -v`
Expected: FAIL — `MetaBlock.__init__` still requires `inv_perm`.

- [ ] **Step 3: Implement**

In `blocks.py`:

```python
    def __init__(self, F, channels, T, perm, conditioner,
                 num_layers, num_heads, expansion, rngs, zero_init=True,
                 use_softplus=True, soft_clip=4.0):
        self.F = F
        self.use_softplus = use_softplus
        self.soft_clip = soft_clip
        self.T = T
        perm = jnp.asarray(perm, dtype=jnp.int32)
        self.perm = Mask(perm)
        self.inv_perm = Mask(jnp.argsort(perm))
```

(rest of the constructor unchanged). In the class docstring, delete the `inv_perm` parameter entry and note under `perm`: "The inverse permutation is derived internally via ``argsort``."

In `model.py:192`, remove `inv_perm=jnp.argsort(perm),` from the `MetaBlock(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/models/tarflow tests/normalizing_flows -v`
Expected: ALL PASS (invertibility tests exercise forward∘inverse with random perms).

- [ ] **Step 5: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/models/tarflow/blocks.py src/gensbi/models/tarflow/model.py tests/models/tarflow/test_blocks_meta.py tests/models/tarflow/test_stability.py
git commit -m "refactor(tarflow): MetaBlock derives inv_perm via argsort(perm)"
```

---

### Task 6: Hoist `_fit_stat` into `models/core/stats.py`

**Files:**
- Create: `src/gensbi/models/core/stats.py`
- Modify: `src/gensbi/models/maf/model.py:207-224` (delete staticmethod, delegate; fix docstring)
- Modify: `src/gensbi/models/tarflow/model.py:288-321` (delete method, delegate)
- Test: `tests/models/core/test_stats.py` (create)

**Interfaces:**
- Produces: `fit_stat(s, example_shape, dtype=None) -> Array` — broadcasts a standardization statistic to `example_shape`; accepts `(dim,)` (reshaped to `(dim, 1, ...)` then broadcast), `(dim, C)`/full shape, `(C,)` (per-channel), or scalar. `dtype` casts before broadcasting (TarFlow writes into existing buffers).
- Consumers: `MAFlow.set_standardization` uses `fit_stat(mean, (self.dim, self.channels)).reshape(-1)`; TarFlow's standardize component uses `fit_stat(mean, self.example_shape, dtype=self.mean[...].dtype)`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/core/test_stats.py`:

```python
import jax.numpy as jnp

from gensbi.models.core.stats import fit_stat


def test_fit_stat_dim_vector():
    out = fit_stat(jnp.arange(3.0), (3, 2))
    assert out.shape == (3, 2)
    assert jnp.array_equal(out[:, 0], jnp.arange(3.0))
    assert jnp.array_equal(out[:, 0], out[:, 1])


def test_fit_stat_full_shape_passthrough():
    s = jnp.arange(6.0).reshape(3, 2)
    assert jnp.array_equal(fit_stat(s, (3, 2)), s)


def test_fit_stat_per_channel():
    out = fit_stat(jnp.array([1.0, 2.0]), (3, 2))   # (C,) with C != dim
    assert out.shape == (3, 2)
    assert jnp.array_equal(out[0], jnp.array([1.0, 2.0]))
    assert jnp.array_equal(out[0], out[1])


def test_fit_stat_scalar_and_dtype():
    out = fit_stat(1.5, (4, 1), dtype=jnp.float32)
    assert out.shape == (4, 1)
    assert out.dtype == jnp.float32
    assert jnp.all(out == 1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/models/core/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: gensbi.models.core.stats`.

- [ ] **Step 3: Implement**

Create `src/gensbi/models/core/stats.py`:

```python
"""Shared standardization-statistic helpers for flow models."""

import jax.numpy as jnp


def fit_stat(s, example_shape, dtype=None):
    """Broadcast a standardization statistic to ``example_shape``.

    Accepted shapes for ``s`` (with ``example_shape = (dim, C, ...)``):

    - ``(dim,)`` — reshaped to ``(dim, 1, ...)`` then broadcast
      (per-dimension stats, the tabular default);
    - ``(dim, C)`` / ``example_shape`` — used as-is;
    - ``(C,)`` — broadcast along the leading axes (per-channel stats);
    - scalar — broadcast everywhere.

    Parameters
    ----------
    s : array-like
        Statistic (mean or std) to fit.
    example_shape : tuple of int
        Target per-example shape, e.g. ``(dim, channels)``.
    dtype : jnp.dtype or None, optional
        If given, cast ``s`` before broadcasting (used when writing into an
        existing buffer). Default is ``None``.

    Returns
    -------
    Array
        ``s`` broadcast to ``example_shape``.
    """
    s = jnp.asarray(s, dtype=dtype)
    if s.ndim == 1 and s.shape[0] == example_shape[0]:
        s = s.reshape((example_shape[0],) + (1,) * (len(example_shape) - 1))
    return jnp.broadcast_to(s, example_shape)
```

In `maf/model.py`: delete the `_fit_stat` staticmethod (lines ~207-212); add `from gensbi.models.core.stats import fit_stat` to the imports; in `set_standardization` replace `self._fit_stat(mean, es)` / `self._fit_stat(std, es)` with `fit_stat(mean, es)` / `fit_stat(std, es)`; in its docstring add the missing per-channel case so it matches TarFlow's wording: "Accepts shapes ``(dim,)`` (broadcast to ``(dim, 1)``), ``(dim, 1)``, ``(C,)`` (per-channel broadcast), or a scalar broadcastable to ``(dim, channels)``."

In `tarflow/model.py`: delete the `_fit_stat` method (lines ~288-293); add the same import; replace `self._fit_stat(mean, self.mean[...].dtype)` with `fit_stat(mean, self.example_shape, dtype=self.mean[...].dtype)` and the `std` line analogously.

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/models/core/test_stats.py tests/models/maf tests/models/tarflow tests/normalizing_flows -v`
Expected: ALL PASS (existing `set_standardization` tests pin behavior for every stat shape).

- [ ] **Step 5: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/models/core/stats.py src/gensbi/models/maf/model.py src/gensbi/models/tarflow/model.py tests/models/core/test_stats.py
git commit -m "refactor(models): hoist duplicated _fit_stat into models.core.stats.fit_stat"
```

---

### Task 7: `patchify_2d` deprecating re-export + serialization docstring path

**Files:**
- Modify: `src/gensbi/recipes/utils.py` (module-level `__getattr__` at end of file)
- Modify: `src/gensbi/utils/serialization.py:11` (docstring)
- Test: `tests/recipes/test_pipeline_utils.py`

**Interfaces:**
- Produces: `gensbi.recipes.utils.patchify_2d` / `.depatchify_2d` resolve again (published docs on `main` teach this import path; gensbi is live on PyPI) with a `DeprecationWarning` pointing to `gensbi.models.core.patching`. One release cycle, then delete.

- [ ] **Step 1: Write the failing test**

Add to `tests/recipes/test_pipeline_utils.py`:

```python
def test_patchify_2d_deprecated_reexport():
    import gensbi.recipes.utils as recipes_utils
    from gensbi.models.core.patching import depatchify_2d, patchify_2d

    with pytest.warns(DeprecationWarning, match="moved to gensbi.models.core.patching"):
        assert recipes_utils.patchify_2d is patchify_2d
    with pytest.warns(DeprecationWarning, match="moved to gensbi.models.core.patching"):
        assert recipes_utils.depatchify_2d is depatchify_2d


def test_recipes_utils_unknown_attribute_still_raises():
    import gensbi.recipes.utils as recipes_utils

    with pytest.raises(AttributeError):
        recipes_utils.does_not_exist
```

(Ensure the file imports `pytest`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/recipes/test_pipeline_utils.py -v`
Expected: FAIL — `AttributeError: module 'gensbi.recipes.utils' has no attribute 'patchify_2d'`.

- [ ] **Step 3: Implement**

Append to `src/gensbi/recipes/utils.py` (the `warnings` import was added in Task 2):

```python
_MOVED_TO_PATCHING = ("patchify_2d", "depatchify_2d")


def __getattr__(name):
    # Deprecated aliases: the functions moved to gensbi.models.core.patching,
    # but main's published docs teach this import path. Keep one release cycle.
    if name in _MOVED_TO_PATCHING:
        warnings.warn(
            f"gensbi.recipes.utils.{name} has moved to "
            "gensbi.models.core.patching; this alias will be removed in a "
            "future release.",
            DeprecationWarning, stacklevel=2)
        from gensbi.models.core import patching
        return getattr(patching, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

In `src/gensbi/utils/serialization.py`, delete the module-docstring line ``See ``docs/superpowers/specs/2026-06-27-safetensors-serialization-design.md``.`` (an installed wheel has no such path and autoapi renders it on the public API page; keep the rest of the docstring).

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n gensbi python -m pytest tests/recipes/test_pipeline_utils.py tests/utils -v`
Expected: ALL PASS.

- [ ] **Step 5: Fast gate + commit**

```bash
mamba run -n gensbi python -m pytest -m "not slow" -q
git add src/gensbi/recipes/utils.py src/gensbi/utils/serialization.py tests/recipes/test_pipeline_utils.py
git commit -m "chore: deprecating patchify_2d re-export; strip internal spec path from serialization docstring"
```

---

### Task 8: Drop `reference/` gitlinks

**Files:**
- Modify: git index (three mode-160000 gitlink entries), `.gitignore`

Fresh clones currently get permanently empty `reference/` dirs and `git submodule update --init` errors ("no submodule mapping found") because there is no `.gitmodules`. Owner decision: drop the gitlinks, keep local checkouts.

- [ ] **Step 1: Remove gitlinks from the index (NOT from disk) and ignore the directory**

```bash
git rm --cached reference/flowjax reference/ml-starflow reference/ml-tarflow
printf '\n# third-party porting references (local checkouts, not part of the repo)\nreference/\n' >> .gitignore
git add .gitignore
```

- [ ] **Step 2: Verify**

Run: `git ls-tree HEAD^{tree} reference/ ; git status --short`
Expected after commit: `git ls-tree HEAD reference/` prints nothing; `git status` shows no `reference/` entries; `ls reference/` still shows the three local checkouts on disk.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: drop reference/ gitlinks (no .gitmodules; empty dirs on fresh clones)"
git ls-tree HEAD reference/   # must print nothing
```

---

### Task 9: Docs — advanced NF page, notebook, wiring

**Files:**
- Create: `docs/advanced/normalizing_flows.md`
- Modify: `docs/advanced/index.md` (toctree), `docs/examples.md` (SLCP section, line ~123-140), `docs/basics/overview.md` (Models section, line ~23-44), `docs/basics/inference.md` (append section), `docs/basics/model_cards.md` (table line ~9-13, bullets ~15-30, descriptions ~32-36)
- Add: `docs/notebooks/slcp_tarflow_nle.ipynb` (currently untracked — would be lost on merge)

- [ ] **Step 1: Create `docs/advanced/normalizing_flows.md`**

````markdown
# Normalizing Flows (experimental)

```{warning}
Discrete normalizing flows are **experimental** in GenSBI. The API is
functional and tested, but may change between releases.
```

Alongside its flow-matching and diffusion methods, GenSBI provides discrete
normalizing flows: conditional density models `q(obs | cond)` whose exact
log-density is available in a **single forward pass**, with no ODE
integration. This makes them natural for likelihood-dominated workflows:

- **NPE** (neural posterior estimation): model `q(theta | x)` directly and
  sample it.
- **NLE** (neural likelihood estimation): model `q(x | theta)`, then sample
  the posterior `p(theta | x_o) ∝ p(theta) q(x_o | theta)` with MCMC —
  practical because the flow's likelihood is exact and cheap to evaluate.

## Models

### MAFlow — Masked Autoregressive Flow

`MAFlow` stacks masked-MLP (MADE) autoregressive layers with affine or
rational-quadratic-spline transformers. It is small, fast to train, and a
solid default for tabular problems up to a few tens of dimensions.

```python
from flax import nnx
from gensbi.models import MAFlow, MAFlowParams

flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_theta, cond_dim=dim_x))
```

### TarFlow — Transformer Autoregressive Flow

`TarFlow` ports Apple's TarFlow/STARFlow transformer autoregressive flow:
stacked causal-attention blocks with alternating token permutations. It
scales to larger problems and supports structured (image) modeled variables
and conditions.

```python
from gensbi.models import TarFlow, TarFlowParams

flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=dim_x, cond_dim=dim_theta,
                             cond="vector", num_blocks=4, layers_per_block=2))
```

The `cond` argument selects the conditioning mechanism: `"bias"` (additive
bias), `"vector"` (per-coordinate condition tokens), or `"image"` (a
patchified image condition attended to as a prefix).

## Training with ConditionalFlowPipeline

`ConditionalFlowPipeline` mirrors the flow-matching pipeline surface
(`sample` / `sample_batched` / `log_prob` / `get_sampler` /
`get_log_prob_fn`), so the diagnostics run unchanged. All tabular tensors
carry the uniform `(B, dim, C)` channel convention (`C = 1` for plain
vectors).

```python
import jax
from flax import nnx
from gensbi.recipes import ConditionalFlowPipeline

pipeline = ConditionalFlowPipeline(
    flow, train_ds, val_ds, dim_obs=dim_theta, dim_cond=dim_x)
pipeline.fit_standardization(theta_train)      # before train()
pipeline.train(nnx.Rngs(0))

x_o = x_observed.reshape(1, dim_x, 1)          # one observation: (1, dim_cond, C)
samples = pipeline.sample(jax.random.PRNGKey(0), x_o)  # (nsamples, dim_theta, 1)
```

Single-observation methods take exactly one observation of shape
`(1, dim_cond, C)`; a batch of conditions goes to `sample_batched`, and a
batched input to a single-observation method raises `ValueError`.

## NLE posterior sampling

For NLE, train the flow with `obs = x`, `cond = theta` (the flow models the
likelihood), then wrap it in
{class}`~gensbi.inference.NLEPosterior`:

```python
from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import MCLMC, NLEPosterior, TemperedSMC

posterior = NLEPosterior(pipeline.ema_model, prior=make_gaussian_prior(dim_theta))
samples = posterior.sample(jax.random.PRNGKey(0), x_o)   # adjusted MCLMC by default
samples, info = posterior.sample(jax.random.PRNGKey(0), x_o,
                                 sampler=TemperedSMC(), return_info=True)
```

The default sampler is adjusted microcanonical Langevin Monte Carlo
(blackjax MCLMC); adaptive tempered SMC is available for multimodal
posteriors. These are convenience samplers — for full control build a
`PosteriorTarget` via `posterior.build_target(x_o)` and run your own
blackjax loop.

## Saving and loading

Flows serialize like any other GenSBI model with the portable safetensors
helpers:

```python
from gensbi.utils.serialization import load_safetensors, save_safetensors

save_safetensors(pipeline.ema_model, "flow.safetensors")
flow2 = MAFlow(params)                     # rebuild the architecture from Params
load_safetensors(flow2, "flow.safetensors")
```

## End-to-end example

See the [SLCP TarFlow NLE notebook](/notebooks/slcp_tarflow_nle) for a
complete workflow: simulate, train a TarFlow likelihood, sample the
posterior with MCLMC, and check calibration.
````

Before committing, sanity-check every import path in the snippets against the code (e.g. `python -c "from gensbi.core.prior import make_gaussian_prior"` in the gensbi env); fix the snippet if a path differs.

- [ ] **Step 2: Wire the toctrees and track the notebook**

- `docs/advanced/index.md`: add `normalizing_flows` as a new line in the toctree (after `custom_models`).
- `git add docs/notebooks/slcp_tarflow_nle.ipynb`.
- `docs/examples.md`, SLCP section (line ~123): add `notebooks/slcp_tarflow_nle` to the SLCP toctree (line ~138-140) and a one-line bullet under the section's "What you'll learn" list: "Neural likelihood estimation with the experimental TarFlow normalizing flow + MCLMC posterior sampling".

- [ ] **Step 3: Short pointers in basics pages (link, don't duplicate)**

- `docs/basics/overview.md`, Models list (after the Flux1Joint bullet, line ~31):

```markdown
- **MAFlow / TarFlow** (experimental): Discrete normalizing flows (masked-autoregressive and transformer-autoregressive) with exact one-pass likelihoods — no ODE integration. Enable NPE and, uniquely, NLE with MCMC posterior sampling. See [Normalizing Flows](/advanced/normalizing_flows).
```

- `docs/basics/inference.md`, append at the end:

```markdown
## Exact-Likelihood Inference with Normalizing Flows (experimental)

GenSBI also ships discrete normalizing flows (`MAFlow`, `TarFlow`) whose
log-density is exact and available in a single forward pass — no ODE solve.
They expose the same `sample`/`sample_batched`/`log_prob` surface through
`ConditionalFlowPipeline`, and enable neural likelihood estimation (NLE)
with MCMC posterior sampling via `gensbi.inference.NLEPosterior`. See the
[Normalizing Flows guide](/advanced/normalizing_flows).
```

- `docs/basics/model_cards.md`:
  - Add a table row: `| **MAFlow / TarFlow** (experimental) | Exact likelihoods, NLE | Low-Medium (MAFlow), Medium-High (TarFlow) | Good | One-pass exact log-prob, no ODE solve, MCMC-ready NLE | Experimental API; autoregressive sampling |`
  - Add a "When to Use" bullet group: "**MAFlow / TarFlow** (experimental): Use when you need exact, cheap likelihood evaluations — likelihood-dominated problems, NLE with MCMC posterior sampling, or fast repeated `log_prob` calls. See [Normalizing Flows](/advanced/normalizing_flows)."
  - Add a Model Description: "**MAFlow / TarFlow** (experimental): Discrete normalizing flows. `MAFlow` is a masked-autoregressive flow (MADE layers, affine or spline transformers) for tabular problems; `TarFlow` is a transformer autoregressive flow (a port of Apple's TarFlow/STARFlow) that scales further and supports image-valued variables and conditions. Both give exact log-densities in one forward pass and integrate with `ConditionalFlowPipeline` and `NLEPosterior`. See [Normalizing Flows](/advanced/normalizing_flows)."

- [ ] **Step 4: Build the docs**

Run (from the repo root): `uv sync --group docs && uv run make -C docs html`
Expected: build succeeds; `docs/_build/html/advanced/normalizing_flows.html` exists; no new warnings about missing toctree entries or broken cross-references to `/advanced/normalizing_flows` or `/notebooks/slcp_tarflow_nle`. (If `uv` groups are not set up in this checkout, use the environment the project's docs CI uses — the gate is a clean `make -C docs html`.)

- [ ] **Step 5: Commit**

```bash
git add docs/advanced/normalizing_flows.md docs/advanced/index.md docs/examples.md docs/basics/overview.md docs/basics/inference.md docs/basics/model_cards.md docs/notebooks/slcp_tarflow_nle.ipynb
git commit -m "docs: advanced Normalizing Flows guide + SLCP NLE notebook; wire NF/NLE into overview, inference, model cards"
```

---

### Final verification (no new code)

- [ ] Full fast suite: `mamba run -n gensbi python -m pytest -m "not slow"` — expected: all green.
- [ ] All four recovery scripts smoke-run on CPU:

```bash
for s in maf_nle_recovery tarflow_nle_recovery tarflow_field_nle_recovery tarflow_image_npe_recovery; do
  mamba run -n gensbi python scripts/$s.py --smoke || echo "FAILED: $s"
done
```

Expected: each prints its smoke-OK line and exits 0.
- [ ] Report to the user: the branch is code-complete; the remaining merge gate is **their full GPU recovery run of all four scripts** — re-examining whether the `num_warmup=500` default now suffices, since the pre-fix `_rescale` bug plausibly caused the earlier 500-vs-2000 discrepancy. Merging to `main` happens only after that gate, via superpowers:finishing-a-development-branch.
