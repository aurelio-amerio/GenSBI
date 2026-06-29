# Uniform `(B, dim, C)` channel convention + conditioner rename/redesign

**Date:** 2026-06-29
**Branch:** maf
**Status:** design approved, pending spec review

## Context

This spec **supersedes Part D** ("channel-convention unification") of
`2026-06-28-flow-pipeline-conformance-and-channels-design.md`, which was
internally contradictory for the `C = 1` case (D1 said "stop squeezing"; D2 said
`example_shape = (dim,)` when `C == 1` — incompatible). Parts A/B/C of that spec
(batch-`>1` warn-and-take-first, kwargs tolerance) are already implemented and
are **unaffected**. The contradiction and its consequences are documented in
`docs/superpowers/handoffs/2026-06-29-channel-convention-C1-inconsistency-handoff.md`.

## Motivation

The library advertises a uniform "expand-dim" convention — every sample-space
tensor carries a channel axis `(B, dim, C)`, with `C = 1` as the default — but
the discrete-flow track (`MAFlow`/`TarFlow` + `ConditionalFlowPipeline`) does
**not** honor it for `C = 1`. It squeezes `(B, dim, 1) → (B, dim)` on the way
into the model (`_squeeze_ch`) and bolts the axis back on with `_expand_dims` on
the way out. So the channel is "convention on paper, collapsed in practice" for
single-channel data, and behaviour differs between `C = 1` and `C > 1`.

The guiding principle for this work, stated by the project owner:

> In this library we **always** carry a channel axis, even when it is size 1, for
> consistency across all kinds of data. There must be **no different behaviour for
> `C = 1` vs `C = N`** — same code path, same shapes, only the channel count
> differs.

A second issue surfaced while designing the fix: TarFlow's conditioners handle
channels inconsistently. The modeled-variable tokenizer folds channels into token
features (the "channel = wider token" picture, à la Flux1), and the image
conditioner does the same via patchification — but the **vector** conditioners
flatten the whole condition through a dense projection, destroying the
per-coordinate structure. Making the vector conditioner *also* tokenize
per-coordinate brings every seam into the same uniform fold and is more
transformer-native. The conditioner classes are renamed at the same time to
remove a `VectorConditioner` double-meaning.

### Mathematical vs implementation reality

A normalizing flow is a bijection on $\mathbb{R}^d$ with a base measure on
$\mathbb{R}^d$; it does not care whether the axes are labelled `(dim,)`,
`(dim, C)`, or `(H, W, C)` — channels are just more coordinates. Carrying a size-1
channel axis is therefore a pure **bookkeeping** choice, numerically free:

- **TarFlow / `VectorTokenizer`**: `tokenize` reshapes `(B, dim, C) → (B, T, F)`
  with `F = block_size · C`. For `C = 1` the element order is identical to the
  current `(B, dim) → (B, T, F)` reshape, and `mean`/`std` of shape `(dim, 1)`
  broadcast to the same values as `(dim,)`. So the internals are **byte-identical**
  for `C = 1`; only a trailing size-1 axis appears on the output.
- **MAF / MADE**: `log_prob` already flattens `(B, dim, C) → (B, dim·C)` via
  `reshape(B, -1)`, so the only change is `sample` always emitting the channel axis.

## Goals

- Make `(B, dim, C)` hold **end-to-end** for every `C ≥ 1`, with `C = 1` carried
  as `(dim, 1)` and never collapsed.
- Eliminate all `C = 1`-vs-`C > 1` branching in the data path: one code path,
  parameterised only by the channel count.
- Bring every TarFlow conditioning seam onto the same channel fold, and rename the
  conditioner classes so each name has exactly one meaning.
- Preserve the correctness invariant: `log_prob` returns `(B,)` — one scalar per
  sample, summed over **all** event coordinates including channels.

## Non-goals

- No condition **compression** inside the flow: a vector conditioner tokenizes the
  condition as given (one token per coordinate). Dimensionality reduction of a
  high-dimensional condition is the **user's** responsibility via an external
  encoder (the VAE pattern in the GW / lensing examples). Consequently **no
  `cond_block_size`** knob.
- No change to the AR factorisation: the SOS input-shift, causal/prefix-LM
  attention masks, and the affine transform are untouched.
- No checkpoint or exact-numerical-identity guarantee. This is a development
  branch; breaking changes are acceptable where mathematically correct. (The
  `C = 1` data-bookkeeping change *is* byte-identical internally; the
  per-coordinate `VectorConditioner` redesign is a genuine architecture change —
  see §3.)
- No `GenerativeMethod`/`ConditionalWrapper` on the flow track; it stays the
  "the flow IS the model" parallel track.

---

## The universal shape contract

The channel axis is **mandatory** on every sample-space tensor. `C = 1` is
`(dim, 1)`, never collapsed. A bare 2-D `(B, dim)` (no channel axis) is
**rejected** with a clear error — strict everywhere.

```
log_prob(x: (B, dim, C_obs),  cond: (B, cond_dim, C_cond)) -> (B,)
sample (key, cond: (B, cond_dim, C_cond))                  -> (B, dim, C_obs)
sample (key, nsamples=n)                                   -> (n, dim, C_obs)   # unconditional
example_shape = (dim, C_obs)                                                    # C_obs = 1 -> (dim, 1)
```

**Why `log_prob` returns `(B,)`, not `(B, 1)`.** A log-probability is a map
*sample-space → ℝ*: one real number per sample. The change-of-variables sum
reduces over **all** event coordinates — `dim` *and* `C` — so the channel is
*consumed* by `log_prob`, not carried through it. The "always carry a channel"
rule applies to data/event tensors (points in sample space), not to scalar
densities; a trailing `1` on `log_prob` would be a vestigial axis with no meaning
(the channels were already summed). `(B,)` is also what the rest of the stack
requires: `gensbi/inference/posterior.py:96` does
`flow.log_prob(x_o[None], theta[None, :])[0]` to extract a **scalar** for
blackjax's `logdensity_fn`; a `(B, 1)` return would yield `(1,)` and misbehave.
Both models already return `(B,)` (TarFlow sums over axes `(1, 2)`; MAF `vmap`s a
scalar), matching the FM pipeline and every standard library.

---

## Part 1 — Rename only (no behaviour change). Lands FIRST, tests green.

Rationale: the channel work introduces data channels (`vec_channels`/
`cond_channels`) and a per-coordinate `VectorConditioner`. Doing the renames up
front means that by the time data channels appear, both `VectorConditioner` and
`channels` each have exactly **one** meaning everywhere — no double-names
downstream. Part 1 is a pure refactor: byte-identical behaviour, all tests pass
after updating to the new names.

### 1a — GenSBI library: conditioner classes + `cond` strings

**Name map (this is a SWAP — order matters):**

| Order | Old | New |
|---|---|---|
| ① first | `VectorConditioner` | `AdditiveBiasConditioner` |
| ② then | `VectorPrefixConditioner` | `VectorConditioner` |
| ③ then | `ImagePrefixConditioner` | `ImageConditioner` |

> ⚠️ `VectorConditioner` is **reused** for a different class. Rename old
> `VectorConditioner → AdditiveBiasConditioner` **first**, *then*
> `VectorPrefixConditioner → VectorConditioner`. The reverse order creates two
> `VectorConditioner`s and corrupts the swap. Use word-boundary matches
> (`VectorPrefixConditioner` does not contain the token `VectorConditioner`).

**`cond=` string values (same step):** `"add" → "bias"`, `"vector_prefix" →
"vector"`, `"image_prefix" → "image"`. The default remains the additive-bias
conditioner (`cond="bias"`).

**Files (from grep):**
- `src/gensbi/models/tarflow/conditioners.py` — the three class definitions.
- `src/gensbi/models/tarflow/model.py` — imports; `make_cond()` dispatch;
  `__post_init__` validation set and the `cond == "image_prefix"` guard;
  docstrings.
- `src/gensbi/models/tarflow/blocks.py` — the conditioner-union docstring on
  `MetaBlock`.
- `tests/models/tarflow/test_conditioners.py`, `test_blocks_meta.py`,
  `test_tarflow.py`, `test_structured_integration.py`, `test_model.py`.

**Behaviour:** byte-identical. At this point the new `VectorConditioner` is still
the *old dense prefix logic* — the per-coordinate redesign happens in Part 3,
operating on the already-renamed class.

### 1b — GenSBI-examples: remove the `channels:` (width) config key

The three tarflow benchmarks (`two_moons/tarflow`, `slcp/tarflow_NLE`,
`slcp/tarflow_NPE`) use `channels:` to mean transformer **width** and back-compute
`num_heads = channels // head_dim` in `build_flow`. Replace the `channels:` key
with an explicit `num_heads:` so the word `channels` no longer means "width"
anywhere (it will mean *data channels* `C`).

**Files:** each benchmark's `config/*.yaml` (all version variants) and its
`train_tarflow_*.py` (`build_flow`).

**Behaviour:** set `num_heads = old_channels // head_dim` so total width
(`head_dim · num_heads`) is unchanged — models identical.

*(No example references the conditioner classes or `cond=` strings — all use the
default conditioner — so 1a does not touch the examples repo.)*

---

## Part 2 — Channel convention in the models

### 2.1 `VectorTokenizer` (`models/core/tokenizers.py`)

Drop the `C == 1` special case:
- `example_shape = (dim, channels)` **always** (so `(dim, 1)` for `C = 1`).
- `detokenize` **always** returns `(B, dim, channels)`.
- `tokenize` unchanged (the reshape already produces identical tokens whether the
  input is `(B, dim)` or `(B, dim, 1)`).

`VectorTokenizer` is used only by TarFlow (confirmed by grep over `src/` and
`tests/`), so no FM/diffusion path is affected.

### 2.2 TarFlow (`models/tarflow/model.py`)

- `example_shape`, `mean`, `std` become `(dim, C_obs)` (from the tokenizer).
- `_ensure_batched` is unchanged and still correct: an unbatched `(dim, 1)`
  (rank 2 = `len(example_shape)`) is promoted; a batched `(B, dim, 1)` (rank 3) is
  not.
- I/O follows the universal contract: `log_prob((B, dim, C_obs)) -> (B,)`;
  `sample(...) -> (B, dim, C_obs)`.

### 2.3 MAF (`models/maf/model.py`)

- `sample` **always** reshapes to `(B, dim, channels)` (drop the
  `if self.channels > 1` guard); for `C = 1` it returns `(B, dim, 1)`.
- `log_prob` is unchanged — it already flattens `(B, dim, C) → (B, dim·C)` and so
  already accepts the channel-carrying input and returns `(B,)`.

After 2.2/2.3, MAF and TarFlow have **identical I/O form**.

---

## Part 3 — Conditioner taxonomy (channel fold + per-coordinate redesign)

`cond_channels` becomes the uniform "condition channel count" that sizes a
conditioner's input. The three conditioners (already renamed in Part 1):

| Class | `cond=` | Mechanism | Channel handling |
|---|---|---|---|
| `AdditiveBiasConditioner` | `"bias"` *(default)* | flatten the whole condition → one `(B, channels)` bias added to every modeled token | **flatten** `(B, cond_dim, C_cond) → (B, cond_dim·C_cond)`, then `Linear(cond_dim·C_cond, channels)`. Unavoidable (single vector out). **`C_cond = 1`: byte-identical to today.** |
| `VectorConditioner` | `"vector"` | **per-coordinate** prefix tokens: `Linear(C_cond, channels)` shared across coordinates → `(B, cond_dim, channels)`, plus per-coordinate positional embeddings; prepended, prefix-LM masked, stripped. `M = cond_dim` | **no flatten** — `Linear` applies on the last (channel) axis directly. |
| `ImageConditioner` | `"image"` | per-patch prefix tokens via `patchify_2d` → `Linear(C·patch², channels)` | per-patch fold `C·patch²` — already implemented; unchanged. |

After this, the two prefix conditioners (`Vector`, `Image`) are structurally
identical: tokenize the condition (per-coordinate or per-patch) → shared per-token
`Linear(feat, channels)` → prefix-concat. Only `AdditiveBias` flattens, where it
is mathematically required.

**`VectorConditioner` redesign specifics:**
- `__init__(cond_dim, cond_channels, channels, rngs)`: `self.proj =
  Linear(cond_channels, channels)`; `self.pos = Param((cond_dim, channels))`.
- `embed(cond: (B, cond_dim, C_cond)) -> (None, proj(cond) + pos[None])`, giving
  `M = cond_dim` prefix tokens of width `channels`. `Linear` operating on the last
  axis *is* the per-coordinate projection — no reshape.
- **Genuine architecture change vs the old dense `Linear(cond_dim, channels·M)`
  prefix** — different parameters, not numerically/checkpoint compatible. This is
  acceptable because (a) dev branch, and (b) **zero blast radius**: no example or
  test trains a prefix-conditioned model (all use `cond="bias"`).
- **`C_cond = 1` caveat (acceptable):** the per-token projection is `Linear(1,
  channels)` — rank-1 (each token is `value · w + b` with shared `w`, identity via
  positional embedding). This is *exactly* how the modeled variable is tokenized at
  `block_size = 1`, so it is consistent with the rest of the model; attention +
  depth provide cross-coordinate mixing.

**Prefix-token count `M`** equals `cond_dim` and is **independent of `C_cond`**
(the channel folds into the projection *input* width, not into extra tokens),
keeping the cond treatment uniform with the obs and image paths. Consequently the
`prefix_tokens` parameter on `TarFlowParams` (which sized the old dense vector
prefix) is **no longer used** by the vector conditioner and is removed.

**`AdditiveBiasConditioner`** likewise gains `cond_channels`: it flattens
`(B, cond_dim, C_cond) → (B, cond_dim·C_cond)` and is sized `Linear(cond_dim·
cond_channels, channels)`. At `C_cond = 1` this is `Linear(cond_dim, channels)`
and the flatten just drops the trailing `1` — numerically identical to today
(the path the SBI benchmarks ride).

**MAF condition** is already handled: `log_prob`/`sample` flatten
`(B, cond_dim, C_cond) → (B, cond_dim·C_cond)` and the MADE conditioner is sized
`cond_dim · cond_channels`.

---

## Part 4 — Pipeline (`recipes/flow_pipeline.py`)

- Retire `_squeeze_ch` as a coercion step. Replace with a **strict rank check**
  that rejects a bare tabular `(B, dim)` (no channel axis) with a clear error, and
  passes channel-carrying input through unchanged.
- Drop the compensating `_expand_dims` on sampler output — the model now returns
  `(nsamples, dim, C_obs)` directly.
- Unify `_single_cond` and `_structured_cond` into one helper that strips the
  leading batch axis and **keeps** the channel (and any further structured axes):
  warn + take-first on a batch axis `> 1` (mirrors the FM pipeline), returning the
  native per-observation shape `(cond_dim, C_cond)` (tabular) or `(H, W, C)`
  (image). The single-observation methods then broadcast to
  `(nsamples,) + that_shape` and hand it to the model, which owns the channel
  mapping.
- `fit_standardization` computes mean/std on the **native** `(N, dim, C)` shape
  (no squeeze). Default `axis=0` → `(dim, 1)` for `C = 1`; `axis=(0, 1)` gives
  per-channel `(1, …, C)`-broadcastable stats for `C > 1` (the `(1, 1, C)` GW
  pattern).
- `structured_obs`/`structured_cond` flags are kept; they select the *tabular*
  `(dim, C)` rank-3 contract vs a genuinely *spatial* `(H, W, C)` one. They no
  longer toggle a squeeze (the channel is always carried).

The exact helper signatures (whether a single `_require_channel(x, name)` plus a
single `_single_obs(x_o)` cleanly replace the four current helpers) are an
implementation detail to be resolved during coding; the contract above is binding.

---

## Part 5 — Tests

- **Tokenizer** (`tests/models/core/test_tokenizers.py`): `C = 1` `example_shape`
  and `detokenize` now `(dim, 1)` / `(B, dim, 1)`.
- **TarFlow** (`tests/models/tarflow/`): `C = 1` `example_shape`/`mean`/`std`
  `(dim, 1)`; `log_prob` in `(B, dim, 1)` → `(B,)`; `sample` → `(B, dim, 1)`. New
  per-coordinate `VectorConditioner` tests (output `(B, cond_dim, channels)`,
  `M = cond_dim`, channel folds on the last axis). Conditioner/`cond`-string
  renames reflected.
- **MAF** (`tests/models/maf/`): `C = 1` `sample` → `(B, dim, 1)`; `log_prob`
  `(B, dim, 1)` → `(B,)`; multichannel round-trip unchanged.
- **Pipeline** (`tests/normalizing_flows/test_flow_pipeline.py`): replace
  `test_squeeze_ch` with a "rejects bare 2-D" test; `C = 1` fixtures move
  `(B, dim)` → `(B, dim, 1)` and conditions to `(1, cond_dim, 1)`; `sample`/
  `log_prob`/`sample_batched` return channel-carrying shapes; unified
  cond-prep keeps the channel.
- **Correctness gate:** `log_prob` returns `(B,)` for every `C`; sampling/density
  round-trips hold; per-channel standardization broadcasts and changes the density
  as expected for `C > 1`.

## Part 6 — Examples (end-to-end verification)

- **MAF benchmarks** (`two_moons/maf_NPE`, `two_moons/maf_NLE`, `slcp/maf_NLE`):
  no model-signature change; confirm they feed `(B, dim, 1)` and inference passes
  channel-carrying observations.
- **TarFlow benchmarks** (the three from Part 1b): add `vec_channels`/
  `cond_channels` as needed (default 1); pass channel-carrying inference
  conditions `(1, cond_dim, 1)`.
- **Gate:** smoke-train all six scripts on CPU (3 MAF + 3 TarFlow, two_moons + slcp) a few
  steps each to confirm end-to-end wiring. Heavier full-training runs remain the
  owner's GPU responsibility.

---

## Sequencing

1. **Part 1** — renames in both repos (behaviour-preserving), tests green.
2. **Part 2** — channel convention in `VectorTokenizer` / TarFlow / MAF.
3. **Part 3** — conditioner channel fold + per-coordinate `VectorConditioner`.
4. **Part 4** — pipeline strictness and unified cond-prep.
5. **Part 5** — test updates to channel-carrying shapes.
6. **Part 6** — examples `vec_channels`/`cond_channels` + smoke-train the six.

## Affected files (anticipated)

- `src/gensbi/models/tarflow/conditioners.py` — Parts 1a, 3.
- `src/gensbi/models/tarflow/model.py` — Parts 1a, 2.2, 3 (conditioner wiring,
  `cond_channels` sizing).
- `src/gensbi/models/tarflow/blocks.py` — Part 1a (docstring).
- `src/gensbi/models/core/tokenizers.py` — Part 2.1.
- `src/gensbi/models/maf/model.py` — Part 2.3.
- `src/gensbi/recipes/flow_pipeline.py` — Part 4.
- `tests/models/{core,tarflow,maf}/…`, `tests/normalizing_flows/test_flow_pipeline.py`
  — Parts 1a, 5.
- GenSBI-examples: `examples/sbi-benchmarks/{two_moons/tarflow, slcp/tarflow_NLE,
  slcp/tarflow_NPE}` configs + scripts — Parts 1b, 6; the three MAF benchmarks —
  Part 6.

## Open questions

None blocking. Implementation resolves the exact pipeline helper signatures
(§Part 4) and the precise `axis` default plumbing for `fit_standardization`.
