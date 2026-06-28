# Flow-pipeline conformance + channel-convention unification

**Date:** 2026-06-28
**Branch:** maf
**Status:** design approved, pending spec review

## Motivation

Two issues surfaced while reviewing the modified `ConditionalFlowPipeline`
(`src/gensbi/recipes/flow_pipeline.py`, the discrete-flow NPE track for
`MAFlow`/`TarFlow`) against the shared flow-matching/diffusion pipeline
(`ConditionalPipeline`):

1. **Behavioural non-conformance.** The flow pipeline's helpers
   (`_single_cond`/`_structured_cond`) warn on a batch axis `> 1` and then take
   the first observation, and the docstrings/tests claim this "mirrors the
   flow-matching pipeline convention." It does **not**: `ConditionalPipeline.sample()`
   warns but passes the *entire* `x_o` through `get_sampler` →
   `_expand_dims(x_o)` with no slicing. With `B > 1` (and `B != nsamples`) the
   model call mis-broadcasts or errors — a guarded latent bug, not a feature.
   The two pipelines genuinely diverge on `B > 1` while advertising that they
   agree. There is also an internal inconsistency: `sample_batched` silently
   swallows `**kwargs`, but `sample`/`get_sampler`/`log_prob` reject them.

2. **The flow track breaks the library-wide channel convention.** The rest of
   the library (datasets, diagnostics, the Flux1 tokenized-transformer FM/diffusion
   models) uses the "expand-dim" convention: observations carry a channel axis
   `(B, dim, C)`, with `C = 1` when there is a single channel. The flow pipeline
   instead squeezes to flat `(B, dim)` and hard-asserts `ch == 1` (`_squeeze_ch`).
   This makes the flow track the odd one out and blocks multi-channel data
   (e.g. the 2-channel gravitational-wave strain in the GenSBI-examples
   `train-gw.py`, which standardizes with a per-channel `(1, 1, 2)` mean) from
   flowing through without manual reshaping.

### Mathematical vs implementation reality (the framing that drove the design)

The channel restriction is an **implementation/convention** choice, not a
mathematical one. A normalizing flow is a bijection on $\mathbb{R}^d$ with a
base measure on $\mathbb{R}^d$; it does not care whether the axes are labelled
`(dim,)`, `(dim, C)`, or `(H, W, C)` — channels are just more coordinates.

- **TarFlow / Flux1**: a channel is naturally a *wider token*. `MetaBlock.proj_in`
  is `Linear(F, channels)` and `proj_out` is `Linear(channels, 2F)`
  (`models/tarflow/blocks.py:138,146`); `F` is already an arbitrary per-token
  feature size (`F = block_size` for the vector tokenizer, `F = C · patch²` for
  the image tokenizer). Channels ride in by enlarging `F`; the projections are
  untouched. The image path *already* does this.
- **MAF / MADE**: no token/projection structure — MADE works on scalar
  coordinates with `(n, n)` autoregressive masks (`models/maf/made.py:133`
  concatenates flat `x` and `cond`). Channels can only mean: flatten
  `(dim, C) → (dim·C,)` and let the channel become additional autoregressive
  coordinates. This is a valid flow (MAF already uses random permutations), so
  flattening is the right, low-cost treatment.

So the structured/image path already handles multi-channel data, including
per-channel standardization (TarFlow's `mean`/`std` are `Mask(example_shape)`).
The only thing missing is carrying the channel through the **tabular vector
path** — which is the gap this spec closes.

## Goals

- Make `ConditionalFlowPipeline` and `ConditionalPipeline` behave identically on
  the two points where they currently diverge (batch `> 1`, kwargs tolerance).
- Make the flow track follow the library-wide `(B, dim, C)` channel convention,
  with `C = 1` as the default, so multi-channel observations and conditions flow
  through without manual reshaping and per-channel standardization works.
- Preserve the core correctness invariant: `log_prob` returns shape `(B,)` — one
  scalar per sample, summed over *all* event coordinates including channels —
  for every `C`.

## Non-goals

- No `GenerativeMethod`/`ConditionalWrapper` added to the flow track (it remains
  the "the flow IS the model" parallel track).
- No change to `proj_in`/`proj_out` or the attention blocks.
- No change to the `_expand_dims`-vs-tokenizer split in the FM/diffusion track.
- No change to model broadcasting math.
- Strict byte/numerical identity for the existing `C = 1` path is **not**
  required (this is a development branch; breaking changes are acceptable when
  mathematically correct). The `C = 1` path should remain mathematically correct
  and is expected to stay numerically close, but exact identity is not a gate.

---

## Part A — shared `ConditionalPipeline`: warn + take-first on batch > 1

**Problem.** `sample()` warns on `B > 1` then passes all conditions through;
`get_sampler`/`get_log_prob_fn` neither warn nor slice. The model call then
mis-broadcasts/errors.

**Change** (`src/gensbi/recipes/conditional_pipeline.py`):

- Add a module-level helper `_single_cond_fm(x_o)`: if `x_o.shape[0] > 1`, emit
  the existing `UserWarning` and slice to `x_o[0:1]` (keep the size-1 batch axis
  that `_expand_dims` and the model expect).
- Call it at the lowest shared point: in `get_sampler` and `get_log_prob_fn`,
  before `_expand_dims`.
- Remove the now-redundant inline warn in `sample()` (the warning lives in
  `get_sampler`, so exactly one fires).
- `sample_batched` is unaffected: it already calls `get_sampler(x_o[0:1], …)`
  per condition, so `B == 1` there — no spurious warnings in the loop.

**Blast radius (intended).** Diffusion and score-matching inherit this through
`ConditionalPipeline`; their latent `B > 1` bug is fixed too. Strictly safer:
the old path errored/misbroadcast, the new one is well-defined.

## Part B — flow pipeline: uniform "accept & warn" kwargs

**Problem.** `sample_batched` swallows `**kwargs`; `sample`/`get_sampler`/
`log_prob` reject them — an inconsistent surface for a drop-in pipeline swap.

**Change** (`src/gensbi/recipes/flow_pipeline.py`):

- Add `_warn_unused_kwargs(kwargs)`: if non-empty, emit a `UserWarning` naming
  the ignored keys (sorted).
- Wire so each unknown key warns exactly once:
  - `get_sampler(x_o, use_ema=True, **kwargs)` and
    `get_log_prob_fn(x_o, use_ema=True, **kwargs)` — warn here.
  - `sample`/`log_prob` — forward `**kwargs` down to those (single call → single
    warn), do not warn themselves.
  - `sample_batched(…, **kwargs)` — warn once itself, and (as today) does not
    forward kwargs into the per-condition `get_sampler` loop, so no `B`×
    duplicate warnings.

## Part C — docstrings + tests for A/B

- Flow pipeline: the "mirrors the flow-matching convention" claim is now true
  (after Part A); tighten wording to "warn + take the first observation in
  `get_sampler`."
- Flow-matching: document warn+take-first on `get_sampler`/`get_log_prob_fn`/
  `sample`.
- New tests:
  - FM (`tests/.../test_conditional_pipeline*`): `get_sampler` and
    `get_log_prob_fn` warn + take-first on `B > 1`; `sample_batched` does not
    warn; `B == 1` is silent.
  - Flow pipeline (`tests/normalizing_flows/test_flow_pipeline.py`): each of the
    four methods warns naming an unknown kwarg; clean calls stay silent.

---

## Part D — channel-convention unification

### D1 — Pipeline carries `(B, dim, C)` end-to-end

`_prep_obs`/`_prep_cond` stop squeezing and stop asserting `ch == 1`; they hand
the model its native channel-carrying input. `_squeeze_ch` is retired as a
shape-coercion step (kept only as an internal rank-check helper if useful). The
model owns the channel → event-shape mapping. The pipeline interface accepts
`(B, dim, C)` (and, for the tabular path, a bare `(B, dim)` which is treated as
`C = 1`).

### D2 — TarFlow (no architectural change)

`VectorTokenizer` gains a `channels=1` parameter:

- `(B, dim, C) → (B, T, F)` with `F = block_size · C`, `T = dim // block_size`.
- `example_shape = (dim, C)` when `C > 1`, `(dim,)` when `C == 1` (keeps the
  current flat path for the common case).
- `proj_in`/`proj_out` untouched — channels enter as wider tokens.
- `mean`/`std` are `Mask(example_shape)`, so per-channel standardization
  broadcasts (the `(1, 1, C)` GW pattern works for free).

The affine transform emits `2F` params (one scale+shift per feature×channel) —
a valid autoregressive coupling.

### D3 — MAF (flatten channels into AR coordinates)

`MAFlow.log_prob`/`sample` accept `(B, dim, C)` and reshape internally to
`(B, dim·C)`:

- Base prior becomes `make_gaussian_prior((dim·C,))`; the `Standardize`
  bijection becomes `dim·C`-dimensional. For `C = 1` these are unchanged.
- `MAFlowParams`/`MAFlow` gain a `channels=1` parameter so the model knows its
  `C` at construction.
- For `C > 1`, channels become additional autoregressive coordinates
  (consistent with MAF's existing random permutations). Document the
  AR-interleave in the model docstring.

### D4 — Per-channel standardization

`fit_standardization` gains an optional reduction-axes argument so a
`(1, …, C)` per-channel mean/std is computable (default reduces over the batch
axis only, as today). The explicit `set_standardization(mean, std)` path already
accepts any broadcastable shape.

### D5 — Correctness invariant (replaces the strict backward-compat gate)

- `log_prob` returns shape `(B,)` — one scalar per sample, summed over **all**
  event coordinates including channels — for every `C`. This is the hard gate.
- The `C = 1` path remains mathematically correct (and is expected to stay
  numerically close to current behaviour), but exact byte/numerical identity and
  checkpoint compatibility are **not** required. All current tests use `C = 1`;
  a small numerical drift for TarFlow from a slightly different embedding path is
  acceptable.

### D6 — `structured_obs`/`structured_cond` flags

Kept. They select the *tabular* `(dim, C)` event shape vs a genuinely *spatial*
`(H, W, C)` one (image tokenizer / `ImagePrefixConditioner`, which already handle
multi-channel). Multi-channel image conditions thus need no reshaping — the
original motivation.

### D-tests

- TarFlow: multi-channel (`C > 1`) vector `log_prob`/`sample` round-trip; output
  shapes `(B, dim, C)`; `log_prob` shape `(B,)`; per-channel standardization
  broadcasts and changes the density as expected.
- MAF: `C > 1` `log_prob` returns `(B,)`; `sample` returns `(B, dim, C)`; the
  flatten round-trips; `C = 1` unchanged.
- Pipeline: `(B, dim, C)` passes through `_prep_*` unsqueezed; `sample`/
  `log_prob`/`sample_batched` return channel-carrying shapes; the old
  `ch == 1` assertion no longer fires for `C > 1`.

---

## Affected files (anticipated)

- `src/gensbi/recipes/conditional_pipeline.py` — Part A.
- `src/gensbi/recipes/flow_pipeline.py` — Parts B, D1, D4.
- `src/gensbi/models/core/tokenizers.py` — Part D2 (`VectorTokenizer` channels).
- `src/gensbi/models/maf/model.py`, `made.py` — Part D3 (channel flatten,
  `channels` param, base/Standardize dim).
- `src/gensbi/models/tarflow/model.py` — Part D2 (`example_shape`, tokenizer
  wiring, standardization shape).
- Tests under `tests/recipes/` (or wherever FM pipeline tests live),
  `tests/normalizing_flows/test_flow_pipeline.py`, `tests/models/maf/`,
  `tests/models/tarflow/`.

## Open questions

None blocking. Implementation will resolve the exact reduction-axes API for
`fit_standardization` and whether `_squeeze_ch` survives as a rank-check helper.
