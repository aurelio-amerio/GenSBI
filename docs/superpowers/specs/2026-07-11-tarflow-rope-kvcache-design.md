# TarFlow: 2D RoPE for image tokens + KV-cached sampling

**Date:** 2026-07-11
**Status:** approved design, pending implementation plan
**Reference:** `reference/ml-starflow` (apple/ml-starflow, `misc/pe.py` + `transformer_flow.py`)

## Goal

Add rotary positional embeddings (RoPE) to `gensbi.models.tarflow` for
image-modeled data, as a faithful port of STARFlow's rotary machinery, and
add a KV-cached sampling path to remove the O(T³) attention cost of the
current per-token full-recompute scan.

## Scope

- RoPE applies to the **modeled image tokens only** (the flow's output — the
  image inference is performed on). Conditioner encodings are untouched.
- STARFlow's latent-VAE / deep-shallow / decoder-finetuning machinery is out
  of scope.
- RoPE for vector-modeled data is out of scope (`use_rope` requires
  `modeled="image"`).
- The KV cache is orthogonal to RoPE and applies to both vector and image
  models, all conditioner types.

## Positional scheme (flux1-style ids, STARFlow 2D schedule)

Every attention slot gets a position id; RoPE rotates q/k by their id.
Rotation by zero is the identity, so a single code path rotates the whole
`[prefix; modeled]` sequence:

- **Modeled image patches:** 2D ids `(px, py)` using STARFlow's normalized
  schedule — `px = arange(h) / sqrt(h·w) · pt_seq_len` (and likewise `py`),
  with `pt_seq_len = img_size // patch_size`. Half of each head's rotary
  dims encode x, half encode y.
- **Prefix (condition) tokens:** ids `(0, 0)` — identity rotation. Token
  identity is carried by the conditioners' existing learned positional
  embeddings, which are unchanged. Precedent: Flux1 sets `txt_ids = 0` for
  the text stream; RoPE attention depends only on relative q−k rotation, so
  an unrotated prefix against rotated image queries is well-posed.

Deliberate deltas from the STARFlow reference (both approved):

1. **No 3D mode.** STARFlow reserves 8 rotary dims per head for a third
   "prefix index" axis because its prefix is text (an ordered sequence).
   Our conditions are generally unstructured parameters θ; rotating prefix
   tokens by index imposes a spurious ordinal metric. Dropping the 3D split
   also removes the `head_dim ≥ 32` constraint that the 8-dim reservation
   would force (`head_dim=16` → 4 freqs per spatial axis is valid; ≥32 is
   recommended for images in the docstrings, not enforced).
2. **Slot-order positions, not permuted.** As in the reference, `freqs_cis`
   is built in raster sequence-slot order and is *not* permuted alongside
   the per-block token flip; flipped blocks effectively see a mirrored
   coordinate grid.

Known non-issue: a prefix token and the top-left patch share the identity
rotation at `(0,0)`; content + learned embeddings disambiguate (same
property holds in Flux1).

Future seams (documented, **not** implemented): a third id axis for
genuinely sequential conditions (STARFlow-style); `(px, py)` ids for
spatially-aligned image conditions.

## Component 1 — RoPE

### New file `src/gensbi/models/tarflow/pe.py`

Faithful JAX port of `ml-starflow/misc/pe.py`, restricted to the paths we
use, with omissions documented in the module docstring:

- `rotate_half(x)`, `apply_rope(t, freqs)` — line-by-line ports.
- `get_positions(...)` — the `'2d'` branch (normalized meshgrid); `'1d'`,
  `'3d'`, and video `duplicate` handling omitted.
- `VisionRotaryEmbedding` — the `'lang'` frequency schedule
  (`1/theta^(2i/dim)`) with `dim = head_dim // 2`, no `latent_len` split,
  no `is_1d` branch, no deprecated checkpoint-compat buffers.

Frequencies are non-trainable constants (stored via the existing `Mask`
pattern or recomputed per call — resolved at implementation; numerically
identical either way, and constant-folded under JIT since all shapes are
static).

### Wiring

- `TarFlowParams`: new fields `use_rope: bool = False`,
  `rope_theta: int = 10000`. Validation in `__post_init__`: `use_rope=True`
  requires `modeled="image"`. Default off → existing configs byte-identical.
- `TarFlow.__init__`: when enabled, constructs one `VisionRotaryEmbedding`
  (mirroring the reference's model-level `feat_rope`) and passes it to every
  `MetaBlock`.
- `MetaBlock`: computes `freqs_cis` for the full `[prefix(M); modeled(T)]`
  slot layout (zeros ids for prefix slots) — the analog of the reference's
  `get_freqs_cis`. When rope is on, the learned `pos_embed` is dropped
  (`None`), matching the reference; the SOS embed is unchanged.
  `freqs_cis` is threaded into every `AttentionBlock` call from
  `_params_core`.
- `AttentionBlock.__call__(x, mask=None, freqs_cis=None)`: rotates q and k
  before `dot_product_attention`. Layout adapted from the reference's
  `(B, h, T, d)` to our `(B, T, h, d)`; numerics identical.
- Conditioners (`AdditiveBiasConditioner`, `VectorConditioner`,
  `ImageConditioner`): no changes.

## Component 2 — KV-cached sampling

The reference `KVCache` is a stateful Python object with dynamic indices —
not JAX-traceable. We port the *behavior* to the standard functional
incremental-decoding pattern (a documented structural deviation):

- Per `MetaBlock`, preallocated k/v buffers of shape
  `(num_layers, B, M+T, num_heads, head_dim)`, carried through `lax.scan`;
  writes via `dynamic_update_slice`.
- `AttentionBlock` gains a decode path: compute q/k/v for the new token
  only; cache **unrotated** k (as the reference does — it applies rope
  after the cache read); each step re-rotates the full cached k with the
  full-length `freqs_cis` and attends under a slot-index length mask
  (`slots ≤ M + i`).
- `MetaBlock.forward` restructured:
  1. embed the condition once (`bias`, `prefix`, `freqs_cis`);
  2. **prefill** — run the M prefix tokens through all layers in one
     parallel pass using the same bidirectional-prefix mask as the training
     path, populating cache slots `[0, M)`;
  3. scan T single-token decode steps (step 0 consumes the SOS embed,
     step i consumes token i−1), applying the affine update per step as
     the current scan body does.
- The current full-recompute scan is **retained** as `_forward_reference`;
  it is the correctness oracle for the cached path, not dead code.
  `TarFlow.sample` uses the cached path.

Complexity at sampling: O(T³) → O(T²) attention work per MetaBlock.

## Error handling

Constructor-time `ValueError` only: `use_rope=True` with
`modeled != "image"`. Everything else is shape-static; no runtime checks
needed. `head_dim` guidance (≥32 recommended for image rope) lives in the
`TarFlowParams` docstring.

## Testing

- **`pe.py` parity:** golden values computed once from the torch reference
  (`misc/pe.py`) and hard-coded as constants in the test (the mamba
  `gensbi` test env has no torch), covering `get_positions` 2d and
  `VisionRotaryEmbedding` frequency/rotation output.
- **Flow properties with rope on:** invertibility
  (`forward ∘ inverse ≡ id`), causality (perturbing token j > i leaves
  z_i unchanged), finite `log_prob`, short training smoke test.
- **KV-cache gate:** cached `forward` ≡ `_forward_reference` to float
  tolerance, per MetaBlock and end-to-end through `TarFlow.sample`, across
  {rope on/off} × {bias, vector, image conditioner} × {vector, image
  modeled}.
- **Regression:** existing fast suite passes untouched with the default
  `use_rope=False`.

Tests run in the mamba `gensbi` env (`JAX_PLATFORMS=cpu` on GPU-less
nodes), per the project's two-env convention.

## Sequencing

Two separable components on one branch, in order:

1. RoPE (`pe.py` port → attention threading → params/wiring → tests).
2. KV-cached sampling (decode path → prefill → scan rewrite → equivalence
   tests), gated on the cached ≡ uncached test.

If the cache fights JAX somewhere unexpected, RoPE can land alone.
