# Flow-pipeline conformance + channel-convention unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the discrete-flow NPE pipeline behave identically to the shared flow-matching pipeline on batch>1 and kwargs handling, and let the flow track carry the library-wide `(B, dim, C)` channel convention (C=1 default) so multi-channel data flows through without manual reshaping.

**Architecture:** Five TDD tasks. Tasks 1–2 are the conformance fixes (shared `ConditionalPipeline` warn+take-first; flow-pipeline accept-&-warn kwargs). Tasks 3–4 give the two flow models channel support (MAF flattens `(dim, C) → (dim·C)`; TarFlow's `VectorTokenizer` folds `C` into token width `F`). Task 5 wires the pipeline to pass `(B, dim, C)` through to channel-aware models and adds per-channel standardization.

**Tech Stack:** JAX, Flax `nnx`, `grain` datasets, `pytest`. Tests force CPU via `os.environ["JAX_PLATFORMS"]="cpu"`.

## Global Constraints

- `log_prob` MUST return shape `(B,)` — one scalar per sample, summed over all event coordinates including channels — for every `C`. This is the hard correctness gate.
- `C = 1` is the default everywhere; the `C = 1` path must remain mathematically correct. Exact byte/numerical identity and checkpoint compatibility are NOT required (development branch; breaking changes acceptable when mathematically correct).
- Do NOT add `GenerativeMethod`/`ConditionalWrapper` to the flow track.
- Do NOT touch `proj_in`/`proj_out` or the attention blocks.
- `TarFlowParams.channels` is reassigned in `__post_init__` to `head_dim * num_heads` (the transformer hidden width). The modeled-vector channel parameter MUST be named `vec_channels`, never `channels`.
- Run the fast suite with the mamba `gensbi` env (not `.venv`).

---

## Task 1: Part A — shared `ConditionalPipeline` warn + take-first on batch > 1

**Files:**
- Modify: `src/gensbi/recipes/conditional_pipeline.py` (add helper; `get_sampler` ~line 223; `get_log_prob_fn` ~line 308; `sample` ~line 275-285)
- Test: `tests/recipes/test_conditional_pipeline.py` (class `TestSampleBatchWarning` ~line 120)

**Interfaces:**
- Produces: module-level `_single_cond_fm(x_o) -> Array` (warns if `x_o.shape[0] > 1`, returns `x_o[0:1]`, else `x_o` unchanged).

- [ ] **Step 1: Write the failing tests** — replace class `TestSampleBatchWarning` (lines ~120-131) with:

```python
class TestSampleBatchWarning:
    def test_sample_batch_xo_warns_and_takes_first(self, pipeline):
        """sample() with batch x_o warns and uses the first condition (no error)."""
        x_o_batch = jnp.zeros((5, dim_cond, 1))  # batch dim > 1
        with pytest.warns(UserWarning, match="batch dimension"):
            s = pipeline.sample(jax.random.PRNGKey(1), x_o_batch, nsamples=4)
        assert s.shape[0] == 4  # one well-defined single-condition draw

    def test_get_sampler_batch_xo_warns(self, pipeline):
        with pytest.warns(UserWarning, match="batch dimension"):
            pipeline.get_sampler(jnp.zeros((5, dim_cond, 1)))

    def test_get_log_prob_fn_batch_xo_warns(self, pipeline):
        with pytest.warns(UserWarning, match="batch dimension"):
            pipeline.get_log_prob_fn(jnp.zeros((5, dim_cond, 1)))

    def test_sample_batched_does_not_warn(self, pipeline, recwarn):
        """sample_batched calls get_sampler with B==1 per condition — no warning."""
        x_o = jnp.zeros((3, dim_cond, 1))
        pipeline.sample_batched(jax.random.PRNGKey(2), x_o, 4,
                                show_progress_bars=False)
        assert not any(
            "batch dimension" in str(w.message) for w in recwarn.list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/recipes/test_conditional_pipeline.py::TestSampleBatchWarning -v`
Expected: `test_get_sampler_batch_xo_warns` / `test_get_log_prob_fn_batch_xo_warns` FAIL (no warning raised today).

- [ ] **Step 3: Add the helper** — in `src/gensbi/recipes/conditional_pipeline.py`, after the imports (after line 50, `import warnings` already present at line 49), add:

```python
def _single_cond_fm(x_o):
    """Reduce a batched conditioning input to a single observation.

    The conditional pipeline's single-observation methods (``sample``,
    ``get_sampler``, ``log_prob``, ``get_log_prob_fn``) condition on ONE
    observation. If ``x_o`` carries a leading batch axis > 1, warn and take the
    first observation (use ``sample_batched`` for many conditions). The size-1
    batch axis is preserved so ``_expand_dims`` and the model broadcast
    correctly.
    """
    x_o = jnp.asarray(x_o)
    n = x_o.shape[0] if x_o.ndim >= 1 else 1
    if n > 1:
        warnings.warn(
            f"x_o has batch dimension {n} > 1. sample()/log_prob() use a single "
            "condition. To use multiple conditions, use sample_batched() instead.",
            UserWarning, stacklevel=3,
        )
        x_o = x_o[0:1]
    return x_o
```

- [ ] **Step 4: Call it in `get_sampler` and `get_log_prob_fn`** — change the `cond = _expand_dims(x_o)` line in BOTH methods (line ~223 and ~308) to:

```python
        cond = _expand_dims(_single_cond_fm(x_o))
```

- [ ] **Step 5: Remove the now-redundant warn in `sample`** — in `sample` (lines ~275-285) delete the `x_o_shape`/`warnings.warn` block so the body becomes:

```python
        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return sampler(key, nsamples)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/recipes/test_conditional_pipeline.py -v`
Expected: PASS (all, including the existing model_extras tests).

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/recipes/conditional_pipeline.py tests/recipes/test_conditional_pipeline.py
git commit -m "fix(recipes): warn+take-first on batch>1 in ConditionalPipeline get_sampler/get_log_prob_fn"
```

---

## Task 2: Part B — flow pipeline uniform "accept & warn" kwargs

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py` (add helper; `get_sampler` line 253; `sample` line 301; `get_log_prob_fn` line 373; `log_prob` line 411; `sample_batched` line 325)
- Test: `tests/normalizing_flows/test_flow_pipeline.py`

**Interfaces:**
- Consumes: `build_pipeline()` test fixture (test file line 37).
- Produces: module-level `_warn_unused_kwargs(kwargs) -> None`.

- [ ] **Step 1: Write the failing tests** — append to `tests/normalizing_flows/test_flow_pipeline.py`:

```python
def test_get_sampler_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.get_sampler(jnp.zeros((1, DIM_COND)), step_size=0.1)


def test_get_log_prob_fn_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.get_log_prob_fn(jnp.zeros((1, DIM_COND)), nsteps=10)


def test_sample_batched_warns_on_unknown_kwarg():
    pipe = build_pipeline()
    with pytest.warns(UserWarning, match="ignores unsupported keyword"):
        pipe.sample_batched(jax.random.PRNGKey(0), jnp.zeros((2, DIM_COND)), 4,
                            solver="dopri5")


def test_known_calls_do_not_warn(recwarn):
    pipe = build_pipeline()
    pipe.get_sampler(jnp.zeros((1, DIM_COND)))
    assert not any(
        "ignores unsupported" in str(w.message) for w in recwarn.list)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/normalizing_flows/test_flow_pipeline.py -k unknown_kwarg -v`
Expected: FAIL — `get_sampler` raises `TypeError` (unexpected kwarg), no warning.

- [ ] **Step 3: Add the helper** — in `src/gensbi/recipes/flow_pipeline.py`, after `_structured_cond` (after line 82), add:

```python
def _warn_unused_kwargs(kwargs):
    """Warn that solver-style kwargs are ignored by the (solver-free) flow.

    The flow pipeline mirrors the flow-matching surface (which accepts
    ``**sampler_kwargs``), but a normalizing flow has no ODE/SDE solver, so
    arguments like ``step_size``/``nsteps``/``solver`` do not apply and are
    silently ignored apart from this warning.
    """
    if kwargs:
        keys = ", ".join(sorted(kwargs))
        warnings.warn(
            f"flow pipeline ignores unsupported keyword argument(s): {keys}. "
            "A normalizing flow has no solver, so these have no effect.",
            UserWarning, stacklevel=3,
        )
```

- [ ] **Step 4: Wire the four entry points** — apply these signature/body edits:

`get_sampler` (line 253): add `**kwargs` and warn first:
```python
    def get_sampler(self, x_o, use_ema=True, **kwargs):
        _warn_unused_kwargs(kwargs)
```
`get_log_prob_fn` (line 373): same:
```python
    def get_log_prob_fn(self, x_o, use_ema=True, **kwargs):
        _warn_unused_kwargs(kwargs)
```
`sample` (line 301): forward kwargs (no warn here):
```python
    def sample(self, key, x_o, nsamples=10_000, use_ema=True, **kwargs):
        return self.get_sampler(x_o, use_ema=use_ema, **kwargs)(key, nsamples)
```
`log_prob` (line 411): forward kwargs:
```python
    def log_prob(self, x_1, x_o, use_ema=True, **kwargs):
        return self.get_log_prob_fn(x_o, use_ema=use_ema, **kwargs)(x_1)
```
`sample_batched` (line 325): warn once, do NOT forward into the loop — change signature and add the warn as the first body line, leaving the existing `x_o = jnp.asarray(x_o)` and loop intact:
```python
    def sample_batched(self, key, x_o, nsamples=10_000, *, use_ema=True,
                       **kwargs):
        _warn_unused_kwargs(kwargs)
        x_o = jnp.asarray(x_o)
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/normalizing_flows/test_flow_pipeline.py -v`
Expected: PASS (new kwarg tests + all existing).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(recipes): uniform accept-and-warn kwargs across flow pipeline methods"
```

---

## Task 3: Part D3 — MAF channel flatten

**Files:**
- Modify: `src/gensbi/models/maf/model.py` (`MAFlowParams` dataclass lines 65-74; `MAFlow.__init__` lines 102-120; `_base` line 122; `log_prob` line 125; `sample` line 151; `set_standardization` line 183)
- Test: `tests/models/maf/test_masked_autoregressive.py`

**Interfaces:**
- Produces: `MAFlowParams(..., channels=1, cond_channels=1)`; `MAFlow.channels`, `MAFlow.cond_channels`, `MAFlow.flat_dim` attributes; `log_prob(x, cond)` accepts `(B, dim, C)` and returns `(B,)`; `sample(...)` returns `(B, dim, C)` when `channels > 1`, else `(B, dim)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/models/maf/test_masked_autoregressive.py` (it already imports `MAFlow, MAFlowParams`; if not, add `from gensbi.models import MAFlow, MAFlowParams` and `from flax import nnx`, `import jax`, `import jax.numpy as jnp`):

```python
def test_maf_channels_one_unchanged():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2))
    x = jnp.zeros((4, 3)); cond = jnp.zeros((4, 2))
    assert flow.log_prob(x, cond).shape == (4,)
    assert flow.sample(jax.random.PRNGKey(0), cond=cond).shape == (4, 3)


def test_maf_multichannel_obs_logprob_and_sample():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, channels=2))
    x = jnp.zeros((4, 3, 2)); cond = jnp.zeros((4, 2))
    assert flow.log_prob(x, cond).shape == (4,)         # scalar per sample
    s = flow.sample(jax.random.PRNGKey(0), cond=cond)
    assert s.shape == (4, 3, 2)                          # channel axis restored


def test_maf_multichannel_cond_flattens():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2,
                               cond_channels=2))
    x = jnp.zeros((4, 3)); cond = jnp.zeros((4, 2, 2))
    assert flow.log_prob(x, cond).shape == (4,)


def test_maf_set_standardization_per_channel_broadcast():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, channels=2,
                               standardize=True))
    flow.set_standardization(jnp.array([1.0, 2.0]), jnp.array([1.0, 1.0]))  # (C,)
    assert flow.log_prob(jnp.zeros((2, 3, 2))).shape == (2,)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/models/maf/test_masked_autoregressive.py -k channel -v`
Expected: FAIL — `MAFlowParams` has no `channels` argument.

- [ ] **Step 3: Add params** — in `MAFlowParams` (after line 74, `zero_init: bool = True`), add:

```python
    channels: int = 1
    cond_channels: int = 1
```

- [ ] **Step 4: Build the chain on flat dims** — replace `MAFlow.__init__` body (lines 102-120) with:

```python
    def __init__(self, params: MAFlowParams):
        rngs = params.rngs
        dim = params.dim
        self.channels = params.channels
        self.cond_channels = params.cond_channels
        flat_dim = dim * self.channels
        flat_cond_dim = params.cond_dim * self.cond_channels
        bijections = []
        for i in range(params.n_layers):
            bijections.append(
                MaskedAutoregressive(flat_dim, flat_cond_dim, params.transformer,
                                     params.nn_width, params.nn_depth, rngs,
                                     zero_init=params.zero_init))
            if i < params.n_layers - 1:
                if params.permutation == "reverse":
                    bijections.append(Permutation.reverse(flat_dim))
                else:
                    bijections.append(Permutation.random(flat_dim, rngs))
        if params.standardize:
            bijections.append(Standardize(flat_dim))
        self.chain = Chain(bijections)
        self.dim = dim
        self.flat_dim = flat_dim
        self.cond_dim = params.cond_dim
```

- [ ] **Step 5: Base over flat dim** — change `_base` (line 122-123):

```python
    def _base(self):
        return make_gaussian_prior((self.flat_dim,))
```

- [ ] **Step 6: Flatten in `log_prob`** — replace the body of `log_prob` (lines 141-149) with:

```python
        base = self._base()
        x = jnp.asarray(x)
        x = x.reshape(x.shape[0], -1)
        if cond is not None:
            cond = jnp.asarray(cond)
            cond = cond.reshape(cond.shape[0], -1)

        def single(x_i, cond_i):
            u, logdet = self.chain.inverse(x_i, cond_i)
            return base.log_prob(u) + logdet

        if cond is None:
            return jax.vmap(lambda xi: single(xi, None))(x)
        return jax.vmap(single)(x, cond)
```

- [ ] **Step 7: Flatten cond + restore channel in `sample`** — replace the body of `sample` (lines 170-181) with:

```python
        base = self._base()
        if cond is not None:
            cond = jnp.asarray(cond)
            cond = cond.reshape(cond.shape[0], -1)
            nsamples = cond.shape[0]
        u = base.sample(key, (nsamples,))

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            x = jax.vmap(lambda ui: single(ui, None))(u)
        else:
            x = jax.vmap(single)(u, cond)
        if self.channels > 1:
            x = x.reshape(x.shape[0], self.dim, self.channels)
        return x
```

- [ ] **Step 8: Broadcast in `set_standardization`** — replace the body of `set_standardization` (lines 188-195) with:

```python
        target = (self.dim,) if self.channels == 1 else (self.dim, self.channels)
        mean = jnp.broadcast_to(jnp.asarray(mean), target).reshape(-1)
        std = jnp.broadcast_to(jnp.asarray(std), target).reshape(-1)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
```

- [ ] **Step 9: Run to verify pass**

Run: `pytest tests/models/maf/test_masked_autoregressive.py -v`
Expected: PASS (new channel tests + all existing).

- [ ] **Step 10: Commit**

```bash
git add src/gensbi/models/maf/model.py tests/models/maf/test_masked_autoregressive.py
git commit -m "feat(maf): support channels by flattening (dim, C) into AR coordinates"
```

---

## Task 4: Part D2 — TarFlow `VectorTokenizer` channels

**Files:**
- Modify: `src/gensbi/models/core/tokenizers.py` (`VectorTokenizer` lines 36-74)
- Modify: `src/gensbi/models/tarflow/model.py` (`TarFlowParams` dataclass lines 111-132; `TarFlow.__init__` tokenizer line 169; `set_standardization` lines 306-307)
- Test: `tests/models/core/test_tokenizers.py` (create if absent) and `tests/models/tarflow/test_blocks_meta.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `VectorTokenizer(dim, block_size=1, channels=1)` with `F = block_size * channels`, `example_shape = (dim,)` if `channels == 1` else `(dim, channels)`; `TarFlowParams(..., vec_channels=1)`; `TarFlow.set_standardization` broadcasts mean/std to `example_shape`.

- [ ] **Step 1: Write the failing tokenizer test** — create `tests/models/core/test_tokenizers.py` (or append if it exists):

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from gensbi.models.core.tokenizers import VectorTokenizer


def test_vector_tokenizer_channels_one_unchanged():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert tok.F == 2 and tok.T == 3 and tok.example_shape == (6,)
    x = jnp.arange(2 * 6).reshape(2, 6).astype(jnp.float32)
    assert jnp.array_equal(tok.detokenize(tok.tokenize(x)), x)


def test_vector_tokenizer_channels_roundtrip():
    tok = VectorTokenizer(dim=6, block_size=2, channels=2)
    assert tok.F == 4 and tok.T == 3 and tok.example_shape == (6, 2)
    x = jnp.arange(2 * 6 * 2).reshape(2, 6, 2).astype(jnp.float32)
    z = tok.tokenize(x)
    assert z.shape == (2, 3, 4)
    assert jnp.array_equal(tok.detokenize(z), x)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/models/core/test_tokenizers.py -v`
Expected: FAIL — `VectorTokenizer` has no `channels` argument.

- [ ] **Step 3: Add channels to `VectorTokenizer`** — replace `__init__` (lines 36-43) and `detokenize` (lines 61-74):

```python
    def __init__(self, dim: int, block_size: int = 1, channels: int = 1):
        if dim % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide dim ({dim})")
        self.dim = dim
        self.channels = channels
        self.F = block_size * channels
        self.T = dim // block_size
        self.example_shape = (dim,) if channels == 1 else (dim, channels)
```

```python
    def detokenize(self, tokens: Array) -> Array:
        """Flatten a token sequence back into a (channelled) vector."""
        B = tokens.shape[0]
        if self.channels == 1:
            return tokens.reshape(B, self.dim)
        return tokens.reshape(B, self.dim, self.channels)
```

(`tokenize` at line 59 is already `x.reshape(x.shape[0], self.T, self.F)` and needs no change — a `(B, dim, C)` input has `B·dim·C = B·T·F` elements.)

- [ ] **Step 4: Run tokenizer test to verify pass**

Run: `pytest tests/models/core/test_tokenizers.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing TarFlow model test** — append to `tests/models/tarflow/test_blocks_meta.py` (import at top if missing: `from gensbi.models import TarFlow, TarFlowParams`, `from flax import nnx`, `import jax`, `import jax.numpy as jnp`):

```python
def test_tarflow_vector_channels_one_unchanged():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              num_blocks=2, head_dim=8, num_heads=2))
    x = jnp.zeros((3, 4))
    assert m.log_prob(x).shape == (3,)
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4)


def test_tarflow_vector_multichannel():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              vec_channels=2, num_blocks=2, head_dim=8,
                              num_heads=2))
    x = jnp.zeros((3, 4, 2))
    assert m.log_prob(x).shape == (3,)                 # scalar per sample
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4, 2)


def test_tarflow_set_standardization_per_channel():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              vec_channels=2, num_blocks=2, head_dim=8,
                              num_heads=2, standardize=True))
    m.set_standardization(jnp.array([1.0, 2.0]), jnp.array([1.0, 1.0]))  # (C,)
    assert m.log_prob(jnp.zeros((2, 4, 2))).shape == (2,)
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/models/tarflow/test_blocks_meta.py -k channel -v`
Expected: FAIL — `TarFlowParams` has no `vec_channels`.

- [ ] **Step 7: Add `vec_channels` param** — in `TarFlowParams` (after line 117, `img_channels: int = 1`), add:

```python
    vec_channels: int = 1
```

- [ ] **Step 8: Pass channels to the vector tokenizer** — in `TarFlow.__init__` (line 169) change:

```python
            tokenizer = VectorTokenizer(params.dim, params.block_size,
                                        params.vec_channels)
```

- [ ] **Step 9: Broadcast in `set_standardization`** — replace lines 306-307:

```python
        self.mean[...] = jnp.broadcast_to(
            jnp.asarray(mean, dtype=self.mean[...].dtype), self.example_shape)
        self.std[...] = jnp.broadcast_to(
            jnp.asarray(std, dtype=self.std[...].dtype), self.example_shape)
```

- [ ] **Step 10: Run to verify pass**

Run: `pytest tests/models/core/test_tokenizers.py tests/models/tarflow/test_blocks_meta.py -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/gensbi/models/core/tokenizers.py src/gensbi/models/tarflow/model.py \
        tests/models/core/test_tokenizers.py tests/models/tarflow/test_blocks_meta.py
git commit -m "feat(tarflow): carry channels as wider tokens via VectorTokenizer vec_channels"
```

---

## Task 5: Part D1 + D4 — pipeline `(B, dim, C)` passthrough + per-channel standardization

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py` (`__init__` lines 113-122; `_prep_obs`/`_prep_cond` lines 124-128; `fit_standardization` lines 194-218; `get_sampler` lines 283-299; `get_log_prob_fn` lines 392-409)
- Test: `tests/normalizing_flows/test_flow_pipeline.py`

**Interfaces:**
- Consumes: `MAFlow` with `channels` (Task 3); `ConditionalFlowPipeline` accept-&-warn kwargs (Task 2).
- Produces: pipeline that passes `(B, dim, C)` through to the model when `ch_obs/ch_cond > 1` (or `structured_*`), returns channel-carrying samples, and `fit_standardization(obs_data, axis=0)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/normalizing_flows/test_flow_pipeline.py`:

```python
def _build_multichannel_pipeline():
    CH = 2
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                               channels=CH, n_layers=4, nn_width=32, nn_depth=2,
                               standardize=True))
    # obs carries a channel axis (N, DIM_OBS, CH); cond stays tabular (N, DIM_COND, 1)
    theta_c = jnp.broadcast_to(_theta[:, :, None], (N, DIM_OBS, CH))
    data = (theta_c, jnp.broadcast_to(_x[:, :, None], (N, DIM_COND, 1)))

    def gen(arr_obs, arr_cond, bs=128):
        idx = grain.MapDataset.source(np.arange(arr_obs.shape[0]))
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
                .map(lambda i: (np.array(arr_obs)[i], np.array(arr_cond)[i])))

    train_ds = gen(data[0][:800], data[1][:800])
    val_ds = gen(data[0][800:], data[1][800:])
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc["val_every"] = 1
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=CH, ch_cond=1, training_config=tc)


def test_multichannel_prep_obs_passthrough():
    pipe = _build_multichannel_pipeline()
    x = jnp.zeros((5, DIM_OBS, 2))
    assert pipe._prep_obs(x).shape == (5, DIM_OBS, 2)   # NOT squeezed


def test_multichannel_sample_and_logprob_shapes():
    pipe = _build_multichannel_pipeline()
    x_o = jnp.zeros((1, DIM_COND, 1))
    s = pipe.sample(jax.random.PRNGKey(0), x_o, nsamples=7)
    assert s.shape == (7, DIM_OBS, 2)                   # channel axis preserved
    lp = pipe.log_prob(jnp.zeros((7, DIM_OBS, 2)), x_o)
    assert lp.shape == (7,)                             # one scalar per sample


def test_fit_standardization_per_channel_axis():
    pipe = _build_multichannel_pipeline()
    obs = jax.random.normal(jax.random.PRNGKey(3), (64, DIM_OBS, 2))
    pipe.fit_standardization(obs, axis=(0, 1))          # per-channel stats
    assert pipe._standardized
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/normalizing_flows/test_flow_pipeline.py -k multichannel -v`
Expected: FAIL — `_prep_obs` squeezes / asserts `ch == 1`.

- [ ] **Step 3: Compute passthrough flags in `__init__`** — in `ConditionalFlowPipeline.__init__` (after line 122, the `self.structured_cond = structured_cond` line), add:

```python
        self._obs_passthrough = structured_obs or ch_obs > 1
        self._cond_passthrough = structured_cond or ch_cond > 1
```

- [ ] **Step 4: Route `_prep_*` on the flags** — replace `_prep_obs`/`_prep_cond` (lines 124-128):

```python
    def _prep_obs(self, x):
        return x if self._obs_passthrough else _squeeze_ch(x)

    def _prep_cond(self, x):
        return x if self._cond_passthrough else _squeeze_ch(x)
```

- [ ] **Step 5: Add `axis` to `fit_standardization`** — change the signature (line 194) and the mean/std lines (213-214). New signature line:

```python
    def fit_standardization(self, obs_data, axis=0):
```

and replace the squeeze/mean/std block (lines 211-215) with:

```python
        obs = jnp.asarray(obs_data)
        if not self._obs_passthrough and obs.ndim == 3:
            obs = _squeeze_ch(obs)
        mean = jnp.mean(obs, axis=axis)
        std = jnp.std(obs, axis=axis)
        std = jnp.where(std < 1e-6, 1.0, std)     # guard zero-variance dims
```

- [ ] **Step 6: Route cond + output on the flags in `get_sampler`** — replace the body after `flow = self.ema_model if use_ema else self.model` (lines 283-299) with:

```python
        if self._cond_passthrough:
            cond = _structured_cond(x_o)             # strip the leading batch axis

            def sampler(key, nsamples):
                cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
                return flow.sample(key, cond=cond_b)  # model owns the output shape
            return sampler

        cond = _single_cond(x_o)                      # (dim_cond,)  [tabular path]

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples, cond.shape[0]))
            samples = flow.sample(key, cond=cond_b)    # (nsamples, dim_obs[, C])
            return samples if self._obs_passthrough else _expand_dims(samples)

        return sampler
```

- [ ] **Step 7: Route cond on the flag in `get_log_prob_fn`** — replace the body after `flow = self.ema_model if use_ema else self.model` (lines 392-409) with:

```python
        if self._cond_passthrough:
            cond = _structured_cond(x_o)             # strip the leading batch axis

            def log_prob_fn(x_1):
                obs = self._prep_obs(x_1)
                cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
                return flow.log_prob(obs, cond_b)
            return log_prob_fn

        cond = _single_cond(x_o)                  # (dim_cond,)  [tabular path]

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)             # (B, dim_obs[, C])
            cond_b = jnp.broadcast_to(cond, (obs.shape[0], cond.shape[0]))
            return flow.log_prob(obs, cond_b)     # (B,)

        return log_prob_fn
```

- [ ] **Step 8: Run the multichannel tests to verify pass**

Run: `pytest tests/normalizing_flows/test_flow_pipeline.py -k multichannel -v`
Expected: PASS.

- [ ] **Step 9: Run the full flow-pipeline + recipes suites for regressions**

Run: `pytest tests/normalizing_flows/test_flow_pipeline.py tests/recipes/test_conditional_pipeline.py -v`
Expected: PASS (C=1 paths unchanged).

- [ ] **Step 10: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(recipes): carry (B, dim, C) through flow pipeline with per-channel standardization"
```

---

## Final verification

- [ ] Run the fast model + recipe suites end-to-end:

Run: `pytest tests/models/maf tests/models/tarflow tests/models/core tests/normalizing_flows tests/recipes -q`
Expected: all PASS.

- [ ] Confirm `log_prob` returns `(B,)` for every channel count exercised (MAF C∈{1,2}, TarFlow C∈{1,2}) — covered by Tasks 3, 4, 5 tests.

## Self-review notes (spec coverage)

- Part A → Task 1. Part B → Task 2. Part C (docstrings/tests) → folded into Tasks 1–2 (tests) and the docstring edits there; the flow-pipeline "mirrors flow-matching" claim is now accurate post-Task 1.
- Part D1 (pipeline `(B,dim,C)`) → Task 5. D2 (TarFlow) → Task 4. D3 (MAF) → Task 3. D4 (per-channel standardization) → Tasks 3/4 (`set_standardization` broadcast) + Task 5 (`fit_standardization axis`). D5 (`log_prob → (B,)` gate) → Global Constraints + every model task. D6 (`structured_*` flags retained, now OR-ed with `ch>1`) → Task 5 Step 3.
- `_squeeze_ch` is retained and now only runs on the `ch == 1` path, where its `ch == 1` assertion is a correct guard (Task 5 Step 4).
