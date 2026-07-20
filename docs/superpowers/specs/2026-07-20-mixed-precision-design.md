# Mixed-Precision Training Design

**Date:** 2026-07-20
**Status:** Approved design, pending implementation plan

## Problem

GenSBI's dtype handling is inconsistent, and in the flux1 family it is actively
wrong:

- `Flux1Params` and `Flux1JointParams` default to `param_dtype=jnp.bfloat16`
  (`models/flux1/model.py`, `models/flux1joint/model.py`), and the models cast
  their inputs to `param_dtype` at the door. There is no separate compute
  dtype anywhere, so weights, activations, model output, loss, gradients,
  AdamW moments, and the EMA all run in **pure bf16**. This is full bf16
  training, not mixed precision.
- The experimental models (PixelDiT, FieldDiT embedders, autoencoders) have
  the same `param_dtype=bf16` defaults.
- `FMLoss` (`flow_matching/loss/fm_loss.py`) and the diffusion losses square
  and mean in whatever dtype the model emits — bf16 for the models above.
- The EMA (`optax.ema`, decay 0.999, `recipes/pipeline.py`) accumulates in the
  param dtype. In bf16 the per-step increment `0.001 · w` is below bf16's
  ~0.4% relative mantissa resolution, so **the EMA accumulator cannot
  integrate small updates at all**. This is the likely root cause of the
  PixelDiT "broken EMA → white-noise samples" mystery (2026-06-14 probe), and
  possibly of FieldDiT's apparent underperformance.
- Simformer and MAF default to fp32 params (correct but with no bf16 speed
  option); TarFlow is fp32 with deliberate fp32 stability paths.
- Already correct: RoPE math is fp32 (`models/flux1/math.py`), and sinusoidal
  timestep-embedding frequencies are built in fp32.

## Goals

1. Master weights, loss, gradients-as-delivered, optimizer state, and EMA in
   fp32, for every model.
2. A per-model compute-dtype knob so the heavy matmuls run in bf16 where safe.
3. Preserve bf16 training speed for the DiT-family models (forward and
   backward GEMMs stay bf16).
4. Fix the EMA accumulation bug as a consequence of (1).

## Non-goals

- Loss scaling (unnecessary for bf16 — same exponent range as fp32).
- fp16 support.
- End-to-end bf16 sampling (solver state stays fp32; can be explored later as
  a wrapper experiment).
- bf16 checkpoint saving (checkpoints double in size for flux1-family; accept
  for now).
- `gensbi.models.healswin` — mirrors the external `heal-swin-nnx` package;
  gets the same treatment as a follow-up in that repo.

## Section 1 — The precision contract

One rule set, applied uniformly to every model:

1. **Master weights fp32.** Every `*Params` dataclass keeps
   `param_dtype: DTypeLike = jnp.float32`. This flips the current bf16
   defaults in flux1, flux1joint, and the experimental models.
2. **Compute dtype is a knob.** Every `*Params` dataclass gains
   `dtype: DTypeLike` — the dtype activations and matmuls run in. Defaults:
   - `jnp.bfloat16`: Flux1, Flux1Joint, Simformer, PixelDiT, FieldDiT,
     autoencoders.
   - `jnp.float32`: MAF, TarFlow (knob present and threaded through, to be
     flipped after dedicated testing — likelihood/log-det computations are
     more precision-sensitive).
3. **fp32 islands** — always fp32 regardless of the knob:
   - Norm layers: LayerNorm / RMSNorm / QKNorm constructed with
     `dtype=jnp.float32` (statistics in fp32; the next Linear re-casts
     activations to bf16, which is the standard pattern).
   - Attention softmax: upcast the `QKᵀ` logits to fp32 before the softmax,
     cast the attention weights back to the compute dtype for the `·V`
     matmul (unless the attention kernel in use already accumulates in fp32
     internally, e.g. cuDNN flash attention).
   - Timestep and sinusoidal embeddings; RoPE (already fp32).
   - **The final output projection runs in fp32** (see contract below).
   - Loss computation (Section 3).
4. **Models emit fp32.** Implemented as the final projection being an fp32
   island — `LastLayer` (and equivalents) constructed with
   `dtype=jnp.float32`, so the bf16 hidden states are promoted and the final
   matmul runs and emits fp32. This is marginally more accurate than a bf16
   matmul followed by an upcast, and the final projection is a negligible
   fraction of total FLOPs (small output dim — a velocity field, not a
   vocabulary). Consequences: the ODE solver, guidance arithmetic, NF
   log-probs, and diagnostics all receive fp32 without any changes.
5. **Inputs stay fp32.** Models stop casting inputs (`obs`, `cond`, `t`) to
   `param_dtype`; layers downcast internally where bf16 compute happens.
6. **Gradients, AdamW moments, EMA: fp32 for free.** JAX gives gradients the
   dtype of the params, and optax builds moments/EMA state from the params.
   With `param_dtype=fp32` the whole optimizer path is fp32 with zero
   pipeline changes. The backward pass still runs its two heavy GEMMs per
   layer in bf16 (autodiff differentiates through the `cast(W_fp32 → bf16)`
   op; the cast's transpose upcasts `dW` to fp32 per layer, an elementwise
   no-cost op), so training speed is preserved.

## Section 2 — Per-model changes

- **flux1 / flux1joint:** `param_dtype` default bf16 → fp32; add
  `dtype=jnp.bfloat16`. Thread both through `layers.py` (MLPEmbedder,
  Modulation, DoubleStreamBlock, SingleStreamBlock, LastLayer) and the model
  files. QKNorm / RMSNorm / LayerNorm → fp32 islands. LastLayer → fp32
  island (emits fp32). Remove input casts to `param_dtype`.
- **simformer:** params already fp32; add `dtype=jnp.bfloat16` and thread
  through `transformer.py` and `model.py` with the same islands.
- **MAF / TarFlow:** add `dtype=jnp.float32` knob and thread through
  `made.py` / `masked_linear.py` and the tarflow blocks. Log-det accumulation
  and the existing softplus / soft-clip stability path stay hard-fp32 even
  when the knob is later flipped to bf16. With the knob at its fp32 default
  this is a pure refactor (bit-identical outputs).
- **Experimental (PixelDiT, FieldDiT, autoencoders):** same treatment as
  flux1 — `param_dtype` default bf16 → fp32, add `dtype=jnp.bfloat16`,
  fp32 islands, final layers emit fp32.

## Section 3 — Loss and pipeline

- **All loss classes upcast to fp32 before reduction.** `FMLoss`
  (`flow_matching/loss/fm_loss.py`), `EDMLoss` and `sm_loss`
  (`diffusion/loss/`), and the NF log-prob losses cast `model_output` and
  targets via `.astype(jnp.float32)` before the squared error / log-prob and
  the mean. Defense-in-depth on top of the models-emit-fp32 contract; also
  protects user-supplied bf16-emitting models.
- **Pipeline needs no optimizer changes.** With fp32 params,
  `nnx.value_and_grad` produces fp32 grads, `optax.adamw` builds fp32
  moments, and `optax.ema` accumulates fp32 — the EMA bug fixes itself with
  no changes to `recipes/pipeline.py` logic.
- **One runtime guard in trainer setup:** when the pipeline is constructed,
  check the model's param-tree dtypes and warn (not error) if master weights
  are not fp32, so bf16 master weights can't be reintroduced silently.
- **Sampling / inference:** fp32 solver state end-to-end automatically (fp32
  inputs in, model emits fp32 out); no changes to `core/ode_solver.py`. The
  network still computes in bf16 internally, which is where the sampling
  speed is, so this loses essentially no performance. The solver update
  `x ← x + dt·v` is the same numerical shape as the EMA bug (small increment
  into a larger accumulator) and must stay fp32.

## Section 4 — Serialization and backward compatibility

- **The safetensors loader casts on restore.** Each loaded tensor is cast to
  the dtype of the corresponding parameter in the target model:
  - Existing bf16 checkpoints (current flux1-family runs) load into the new
    fp32 models via a lossless upcast — old checkpoints stay usable.
  - An fp32 checkpoint loads into a model deliberately built with
    `param_dtype=bf16` (lossy downcast, the user's explicit choice).
- **Checkpoint size doubles** for the flux1-family models (bf16 → fp32
  storage). Accepted; a `save_dtype` option can be added later if disk
  becomes a problem.
- Orbax training checkpoints restore into a freshly constructed model (now
  fp32 params); apply the same cast-to-target-dtype rule on dtype mismatch.

## Section 5 — Testing and validation

- **Unit tests per model** (flux1, flux1joint, simformer, MAF, TarFlow,
  PixelDiT, FieldDiT, autoencoders): with `dtype=bf16` (or the knob at its
  default for the NF models), assert:
  - (a) every leaf of the param state tree is fp32;
  - (b) forward output dtype is fp32;
  - (c) gradients from a dummy loss are fp32;
  - (d) AdamW moment and EMA state trees are fp32.
- **Numerical closeness:** small instance of each bf16-default model, forward
  pass with `dtype=fp32` vs `dtype=bf16` at identical weights, relative
  tolerance ~1e-2. Catches plumbing mistakes (a cast in the wrong place
  produces garbage, not rounding noise).
- **EMA regression test (bug-shaped):** apply many small (0.1%-scale) param
  updates and assert the EMA accumulator integrates them. Fails on today's
  bf16 configuration, passes after the fix.
- **MAF / TarFlow pure-refactor invariant:** with the knob at the fp32
  default, outputs bit-identical to current code.
- **GPU validation gate (user-run):**
  - Rerun the PixelDiT GRF probe with `use_ema=True` — if the white-noise
    EMA mystery was this bug, sampling should now produce structure.
  - A flux1 two-moons (or similar) sanity run confirming mixed-precision
    training converges as before or better.

## Follow-ups (out of scope)

- Flip MAF / TarFlow `dtype` to bf16 after dedicated stability testing.
- Same treatment for `heal-swin-nnx` (external repo).
- Optional `save_dtype` for smaller checkpoints.
- Optional end-to-end bf16 sampling experiment (wrapper, not core).
