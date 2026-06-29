# Handoff: `(B, F, C)` channel convention is not uniform for `C = 1`

**Date:** 2026-06-29
**Branch:** maf
**Status:** investigation complete — needs a fresh brainstorming session to revise the spec and re-implement
**Repos involved:**
- `GenSBI` (`/lhome/ific/a/aamerio/data/github/GenSBI`) — library, where the fix lives
- `GenSBI-examples` (`/lhome/ific/a/aamerio/data/github/GenSBI-examples`) — the example training scripts that triggered the investigation

## Why this handoff exists

While preparing to fix the normalizing-flow example training scripts (MAF + TarFlow,
two_moons + slcp) for the new `(B, F, C)` data convention, we discovered that the
governing design doc

> `docs/superpowers/specs/2026-06-28-flow-pipeline-conformance-and-channels-design.md`

**contradicts itself on the `C = 1` case**, and the implementation followed the
conservative half. As a result the advertised uniform `(B, F, C)` convention does
**not** hold end-to-end for single-channel data — it silently collapses to `(B, F)`.
We are stopping here so a fresh session can revise the spec and conform the library.

## The contradiction (verbatim from the spec)

- **Part D1 (pipeline)** says:
  > "`_prep_obs`/`_prep_cond` **stop squeezing** and stop asserting `ch == 1`; they
  > hand the model its **native channel-carrying input**. `_squeeze_ch` is retired
  > as a shape-coercion step ... The pipeline interface accepts `(B, dim, C)`."

- **Part D2 (TarFlow tokenizer)** says:
  > "`example_shape = (dim, C)` when `C > 1`, **`(dim,)` when `C == 1`** (keeps the
  > current flat path for the common case)."

These are incompatible for `C = 1`. If the model's *native* event shape for `C = 1`
is flat `(dim,)` (D2), then the pipeline **must** squeeze `(B, dim, 1) → (B, dim)` to
feed it — which is exactly what D1 says it must stop doing.

## What was actually implemented (verified in code)

The implementer resolved the conflict in favour of **D2** (flat `C = 1`), keeping the
squeeze for the single-channel path:

- `src/gensbi/models/core/tokenizers.py:51`
  ```python
  self.example_shape = (dim,) if channels == 1 else (dim, channels)
  ```
  → for `C = 1` the TarFlow event shape is flat `(dim,)`, so `mean`/`std`
  (`tarflow/model.py:209-210`) are `(dim,)` too.

- `src/gensbi/recipes/flow_pipeline.py:148-152`
  ```python
  def _prep_obs(self, x):
      return x if self._obs_passthrough else _squeeze_ch(x)
  def _prep_cond(self, x):
      return x if self._cond_passthrough else _squeeze_ch(x)
  ```
  `_obs_passthrough = structured_obs or ch_obs > 1`. So on the **default `ch = 1`
  path** the pipeline still calls `_squeeze_ch` (`flow_pipeline.py:17-28`), collapsing
  `(B, dim, 1) → (B, dim)` — contradicting D1's "stop squeezing".

- On the sampling side the pipeline then re-adds the axis via `_expand_dims(samples)`
  (`flow_pipeline.py:330`) so outputs come back as `(nsamples, dim, 1)`. So the
  channel axis is squeezed on the way in and bolted back on the way out.

**Verdict:** the implementation is faithful to **D2** and to **D1 for `C > 1`**, but
violates D1's universal "stop squeezing." "Stop squeezing / native channel-carrying
input" was only realized for `C > 1` / structured. The `(B, F, C)` convention does
**not** hold end-to-end for `C = 1`. This was not a sloppy implementation of a coherent
spec — the spec asked for two incompatible things and the safe one was chosen.

## Data reality for the example tasks

The investigation also confirmed the data side (so the next session doesn't re-derive it):

- `sbibm-jax` conditional collate adds a trailing channel axis:
  `src/sbibm_jax/data/process.py:101-102` →
  `theta = ...[..., None]`, `x = ...[..., None]`.
  So batches are `(B, dim, 1)` for both θ and x. **two_moons and slcp are `C = 1`.**
- Through the current pipeline, the `ch = 1` squeeze hands the models flat `(B, dim)`,
  so the existing example scripts *do* run for `C = 1` — the bug is the convention
  inconsistency, not a crash.

## Model signature changes already in place (context for the examples fix)

- **MAF** (`src/gensbi/models/maf/model.py`): `MAFlowParams` gained `channels=1`,
  `cond_channels=1`. `log_prob` reshapes `(B, dim, C) → (B, dim·C)` (so it already
  accepts `(B, dim)` and `(B, dim, 1)`); `sample` returns `(B, dim)` for `channels=1`,
  `(B, dim, C)` for `channels>1`.
- **TarFlow** (`src/gensbi/models/tarflow/model.py`): now uses the Flux1 convention —
  specify `head_dim` + `num_heads`; total width `channels = head_dim * num_heads` is
  **derived** in `__post_init__` (`tarflow/model.py:148`). Separately, `vec_channels=1`
  is the *data* channel count fed to `VectorTokenizer`.
  - **Naming collision to fix in the examples:** the example tarflow configs still use
    `channels:` to mean the transformer **width** and back-compute
    `num_heads = channels // head_dim` in `build_flow`. Under the new convention the
    configs should specify `head_dim` + `num_heads` directly (and `channels` as a
    config key should go away, since the word now means data channels `C`).
    Affected: `two_moons/tarflow`, `slcp/tarflow_NLE`, `slcp/tarflow_NPE` (configs +
    each script's `build_flow`). The 3 MAF scripts need no signature change.

## Recommended fix direction (to be decided in the fresh brainstorm)

Make the `(B, F, C)` convention genuinely uniform by keeping `C = 1` as a size-1 axis
everywhere (i.e. revise **D2**, tighten **D1**):

1. `VectorTokenizer.example_shape = (dim, channels)` **always** → `C = 1` is `(dim, 1)`.
2. TarFlow `mean`/`std` become `(dim, 1)`; `log_prob`/`sample` carry `(B, dim, 1)` for
   `C = 1`. (`_ensure_batched` promotes `(dim,1)→(1,dim,1)` correctly.)
3. MAF `sample` returns `(B, dim, 1)` for `channels=1` (its `log_prob` already flattens).
4. Pipeline genuinely stops squeezing on the `ch = 1` path: retire `_squeeze_ch` as a
   coercion step (keep at most as an optional rank check) and drop the compensating
   `_expand_dims` on output — pass `(B, dim, 1)` straight through, as D1 originally
   intended.
5. Update the `C = 1` unit tests from `(B, dim)` to `(B, dim, 1)` shapes. The spec
   already states exact `C = 1` byte/shape identity is **not** a gate on this dev branch.

**Why this is low-risk:** for `C = 1`, `T = dim // block_size` and `F = block_size · C`
are unchanged, and the base log-prob sums over the same elements — the change is purely
trailing-axis bookkeeping (numerically free), and it makes per-channel standardization
`(1, 1, C)` broadcast uniformly.

### Open questions for the fresh brainstorm

- Does any **non-NF / FM-diffusion** code path consume `VectorTokenizer` with `C = 1`
  and rely on the flat `(dim,)` `example_shape`? Grep `VectorTokenizer` usage before
  committing to "always `(dim, channels)`." (TarFlow is the only confirmed user so far.)
- Should `_squeeze_ch` survive at all, or be deleted entirely?
- Confirm whether the bare-`(B, dim)` tabular input (no channel axis) should still be
  accepted as a convenience alias for `C = 1`, or whether `(B, dim, 1)` becomes the
  only accepted form.
- Verification plan: run affected GenSBI unit tests (tokenizer / MAF / TarFlow /
  flow pipeline), then smoke-train the 6 example scripts (3 MAF + 3 TarFlow,
  two_moons + slcp) a few steps each to confirm end-to-end wiring.

## Key files

| Concern | Path |
|---|---|
| Governing (contradictory) spec | `GenSBI/docs/superpowers/specs/2026-06-28-flow-pipeline-conformance-and-channels-design.md` |
| Tokenizer `example_shape` | `GenSBI/src/gensbi/models/core/tokenizers.py:51` |
| TarFlow mean/std + `_ensure_batched` | `GenSBI/src/gensbi/models/tarflow/model.py:165-217, 287-311` |
| TarFlow param convention (`channels` derived) | `GenSBI/src/gensbi/models/tarflow/model.py:111-148` |
| MAF flatten/sample | `GenSBI/src/gensbi/models/maf/model.py:104-206` |
| Pipeline squeeze/passthrough | `GenSBI/src/gensbi/recipes/flow_pipeline.py:17-28, 148-152, 330` |
| Data collate adds `[..., None]` | `sbibm-jax/src/sbibm_jax/data/process.py:101-102` |
| Example tarflow configs/scripts (naming collision) | `GenSBI-examples/examples/sbi-benchmarks/{two_moons/tarflow, slcp/tarflow_NLE, slcp/tarflow_NPE}` |
| Example maf scripts (no signature change) | `GenSBI-examples/examples/sbi-benchmarks/{two_moons/maf_NPE, two_moons/maf_NLE, slcp/maf_NLE}` |
