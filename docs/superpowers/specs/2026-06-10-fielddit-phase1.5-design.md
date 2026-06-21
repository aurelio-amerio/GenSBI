# FieldDiT Phase 1.5 — Harden, Wire, Prove (Design)

**Date:** 2026-06-10
**Branch:** `FieldDiT` (continues Phase 1, un-merged)
**Predecessors:**
- Design spec: `docs/superpowers/specs/2026-06-09-fielddit-design.md`
- Phase-1 plan: `docs/superpowers/plans/2026-06-09-fielddit-phase1-model.md`
- Handoff: `docs/superpowers/notes/2026-06-09-fielddit-phase1-handoff.md`

## 1. Goal and scope

Phase 1 delivered a *well-formed* FieldDiT: correct shapes, finite, exactly zero
at init, differentiable. Phase 1.5 turns it into a *trainable and
verified-alive* model by resolving the handoff's two gates (B1, B2), the cheap
hardening notes (B5), and the actionable findings of a post-handoff code
review.

**In scope:** model-internal hardening (W1), a field-shaped conditional
pipeline (W2), learning-gate and backfill tests (W3). All work lands in GenSBI
on the `FieldDiT` branch.

**Out of scope (recorded deferrals, §6):** GRF 256² validation (lives in
GenSBI-examples, already started there); the B3/B4 design studies (they need a
trained model — they become GRF ablations); CFG / null-conditioning; Phase-2
image conditioning; mixed precision.

## 2. W1 — Model hardening

All changes are in `src/gensbi/experimental/models/fielddit/` unless noted.
They alter parameters/numerics, so they land **before** anything trains.

### 2.1 GraphDef hygiene (review finding, major)

Storing the `FieldDiTParams` dataclass on the module makes the nnx GraphDef
unhashable (it holds `nnx.Rngs`, a `list`, and derived `jnp` arrays) and
never-equal across instances: `nnx.jit` retraces per model instance (bites the
pipeline's EMA/eval-model pattern) and functional `jax.jit` with a static
graphdef is impossible. Fix:

- `FieldDiT.__init__` copies the primitive fields `__call__` needs
  (`field_shape`, `in_channels`, `cond_dim`, `use_cond_summary_in_vec`,
  `guidance_embed`, `param_dtype`, `token_grid`, new flags) onto the module as
  plain attributes; `self.params` is no longer stored.
- `obs_ids` / `cond_ids` construction moves out of
  `FieldDiTParams.__post_init__` into `FieldDiT.__init__`. The ids are stored
  in a dedicated `nnx.Variable` subclass (e.g. `RopeIds`) so they are
  filterable and immune to blanket state-dtype casts (a `tree.map` f32↔bf16
  cast over the state must not corrupt int32 rope ids).
- `axes_dim` becomes a tuple after `__post_init__` normalization.
- `rngs` **stays** in the dataclass (consistent with `Flux1Params`); the
  advances-on-construction semantics (two `FieldDiT(params)` calls from the
  same params object yield different weights) are documented in the
  `FieldDiTParams` docstring.

### 2.2 Transformer→decoder boundary norm (review finding)

Flux1 leaves the residual stream through `LastLayer` (norm + projection);
FieldDiT's `Untokenizer.proj` currently projects the raw stream straight into
the conv decoder, and the stream's magnitude grows with transformer depth.
Add a `LayerNorm` before `Untokenizer.proj`. Output-zero-at-init is unchanged
(it is guaranteed by the decoder's zero-init `conv_out`).

### 2.3 Loud failures instead of silent wrongness

- `conditioned=False` (or non-True) in `FieldDiT.__call__` raises
  `NotImplementedError`. Today it is silently ignored: an "unconditional" CFG
  pass would quietly return the conditional output. CFG is deliberate later
  work (§6).
- Guard `obs.shape[1:3] == field_shape` and `obs.shape[-1] == in_channels` at
  the top of `__call__`. Today a wrong spatial size fails deep inside
  attention with a cryptic broadcast error (verified).
- `ScalarCondEmbedder` raises on 2D input when `cond_in_channels != 1`
  (today: documented-only caveat, cryptic downstream error).

### 2.4 Numerics

Compute `timestep_embedding(t, 256)` in float32 and cast the resulting `vec`
to the model dtype afterwards. Today `t` is cast to bfloat16 *first*,
quantizing t to ~0.004 resolution (inherited from Flux1; fixed here locally,
Flux1 itself is out of scope).

### 2.5 Conditioning symmetry flag (new, from the ResUViT comparison)

New `FieldDiTParams.cond_modulates_encoder: bool = False`. When `True`, the
encoder receives the full modulation vector `vec` (time + cond summary
+ guidance) instead of `time_vec`, making encoder and decoder modulation fully
symmetric.

Rationale recorded here as the load-bearing reason for the default: a
condition-free encoder computes identical features for the conditional and
unconditional branches of a CFG sampling step (shared `x_t`, `t`), so the
encoder pass and its skips can be shared; it also keeps the encoder a pure
noisy-field feature extractor for Phase-2 read-only conditioning. The flag is
the escape hatch that turns handoff risk **B4** ("does coarse + decoder-FiLM
conditioning suffice?") into a switchable GRF ablation instead of an
architectural bet. The flag does not change the zero-at-init property
(modulation linears are zero-init).

Context from the reference comparison (`reference/bayesflow/networks/subnets/
unet/resuvit.py`): ResUViT injects cond by channel-concat at the input — for
scalar/vector θ that is a weaker mechanism (a constant-plane additive bias at
stage 0) than FieldDiT's per-stage FiLM + cond tokens in joint attention;
concat's real strength (spatially-resolved field-shaped cond) is exactly the
deferred Phase-2 case. Moving away from concat is deliberate and correct.

### 2.6 Defaults and hygiene

- `theta: Optional[int] = None`, derived in `__post_init__` as
  `min(10 * (n_obs_tokens + cond_dim), 10_000)` (rule of thumb: 10× token
  count, capped at 10k). An explicitly passed `theta` always wins.
- Resolve the stale `model.py:61` comment with the decided position: the rope
  semantic axis stays unrotated/inactive in Phase 1 (cond uses learned
  absolute embeddings, not rope, so the zero-id "collision" with obs token
  (0,0) is benign — same situation as Flux1 txt/img); the id scheme
  (semantic dims, `init_ids_1d`/`init_ids_2d` axis order, `axes_dim` split)
  is redesigned wholesale in Phase-2 co-tokenization, not piecemeal now.
- Rename `ObsDecoder`'s `in_channels` parameter to `out_channels` (it is the
  output channel count of `conv_out`).

## 3. W2 — Field pipeline (handoff B2)

New code in `src/gensbi/experimental/recipes/` (existing package, currently
holds `vae_pipeline.py`); one shared-util change in `src/gensbi/core/prior.py`.

### 3.1 Shape-generic Gaussian prior

`make_gaussian_prior` currently hard-codes rank-2 event shapes, and
`make_gaussian_prior(H, W, C)` today silently reads `C` as the prior *mean*.
Generalize to arbitrary-rank event shapes
(`Independent(Normal(zeros(shape)), len(shape))`) with an API that makes the
positional-mu mistake impossible (exact signature decided in the plan;
backward compatible for existing `(dim, ch)` callers).

### 3.2 `FieldConditionalPipeline`

Subclass of `ConditionalPipeline`:

- `event_shape = (H, W, C)` from the model's field shape; prior and path built
  with it. `prepare_batch` then yields field-shaped `x_0`, and sampling
  returns `(nsamples, H, W, C)`.
- Skips obs-id resolution entirely (FieldDiT builds rope ids internally);
  passes `cond` through raw (the embedder handles `(B, k)` / `(B, k, c)`).
- Wraps the model in a new `FieldConditionalWrapper`: event-ndim-aware input
  expansion (compare against the known event rank) instead of the generic
  `ndim < 3` heuristic, which silently misreads unbatched field-shaped `obs`
  and unbatched `(k, c)` cond; no id expansion.

### 3.3 Pipeline tests

Tiny config: pipeline construction, one training step (loss is finite scalar,
params change), sampling shape `(nsamples, H, W, C)`.

## 4. W3 — Learning gates (handoff B1) and test backfill

In `tests/experimental/models/fielddit/` (plus pipeline tests above).

- **Gate 1 — aliveness:** after one optimizer step on a trivial objective, the
  output is no longer identically zero, and gradients are nonzero for
  representative encoder / core / decoder-block parameters (not just
  `conv_out`). Rationale: at init, zero-init gates make 149/151 parameter
  gradients identically zero by design; no test at init can distinguish a live
  conditioning path from a dead one.
- **Gate 2 — tiny overfit:** drive the loss down on a handful of
  (field, cond) pairs; assert the loss drops materially **and** that different
  conds produce different outputs after training (kills the dead-conditioning
  failure mode).
- **Core de-identity test:** with randomized modulation parameters,
  `MMDiTCore` output (a) changes when `cond` changes and (b) is not
  permutation-equivariant in obs tokens (rope active). Today the core is
  bit-exactly the identity at init, so the existing core tests exercise no
  attention/rope/cond path (verified).
- **Backfill:** non-square field, `patch_size=1`, single-level encoder
  (`encoder_widths` of length 2), `nnx.split`/`merge` + `nnx.jit` round-trip.
  All four verified to work today; the tests pin them.
- **Opt-in realistic-size smoke:** instantiate a 256² / hidden-768 config and
  run one forward pass; record the derived token count and peak memory.
  Skipped by default (env-flag opt-in), not part of CI.

## 5. Execution order

1. **W1** hardening (changes params/numerics — must precede training).
2. **W3 backfill + core de-identity test** (locks W1 in).
3. **W2** field pipeline.
4. **W3 gates** (need the pipeline's loss/optimizer wiring) + opt-in smoke.

Verification at each step: `JAX_PLATFORMS=cpu uv run pytest
tests/experimental/models/fielddit/ tests/recipes/` (no regressions; 216
passed pre-existing baseline).

## 6. Recorded deferrals

| Deferred item | Trigger to revisit |
|---|---|
| f32 master weights + bf16 compute (today: single dtype, bf16 default; weight updates ≲4e-3 relative underflow bf16) | Before the first real GRF training run; earlier if the W3 overfit gate stalls at toy scale |
| CFG / null-conditioning, batch-1 cond broadcast in `MMDiTCore` | Phase-2 sampling work |
| `theta` tuning, B3 (GroupNorm vs field statistics), B4 (fine-scale conditioning response), encoder-modulation flag ON vs OFF | GRF ablations in GenSBI-examples |
| Rope id-scheme unification (`init_ids_1d` vs `init_ids_2d` semantic-axis order, `axes_dim` split) | Phase-2 co-tokenization design |
| Flux1's own shared warts (bf16 timestep embedding, unhashable graphdef) | Separate issue, stable-code change policy |

## 7. Non-goals

No public-API renames beyond `ObsDecoder.out_channels`; no changes to stable
(`gensbi.models.flux1`, `gensbi.recipes.conditional_pipeline`) behavior; no
training-performance optimization; no GRF data or example code in this repo.
