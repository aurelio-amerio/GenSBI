# TransformerFlow stability port (softplus + soft_clip + fp32)

**Date:** 2026-06-25
**Status:** Design — approved, pending spec review
**Scope:** `src/gensbi/normalizing_flows/transformer_flow/` only (`blocks.py`, `model.py`)

## Problem

After the v1→v2 TransformerFlow change (output-shift → STARFlow SOS input-shift), the
two-moons NPE example diverges to `loss=nan` mid-training (~17k steps) where the v1 model
converged. Root cause (confirmed by reading both reference repos + a CPU repro):

- v1 (faithful to apple/ml-tarflow) pins the **first token of every block to identity**
  (`a=b=0` via the output-shift's leading zeros). No free scale on that token.
- v2 (apple/ml-starflow SOS) replaces that with a **learned, batch-independent affine**
  on the first token. Across the 8 flip-alternating blocks this stacks into a set of
  **redundant, unconstrained per-block log-scales** on each dimension — a near-flat loss
  direction that Adam drifts along until the **unbounded `exp` affine overflows fp32**
  → `inf` → `NaN`. The pipeline has no NaN recovery (`optax.adaptive_grad_clip(10.0)`
  is permissive and does nothing once values are already non-finite), so AdamW's moments
  are poisoned permanently.

Crucially, **STARFlow never runs SOS + bare `exp`.** Its released configs
(`reference/ml-starflow/configs/*.yaml`) pair `sos: 1` with `use_softplus: 1` and
`soft_clip: 4` (video also adds `grad_clip`/`grad_skip`). We ported the SOS shift but
none of the guards. This spec ports the model-level guards.

## Decisions (from brainstorming)

- **Default ON.** The SOS shift is unconditional, so the model is unstable by default;
  the stabilizers must be on by default. (Contrast MAF, which is stable by default — see
  Out of scope.)
- **Model-scoped only.** No changes to the shared pipeline. softplus+soft_clip prevent
  overflow at the source, so no pipeline NaN/grad-skip guard is added.
- **Keep `exp` as a non-default fallback** (`use_softplus=False`) for faithful-TarFlow
  reproduction and a first-class stability regression test.
- **MAF out of scope** — separate track, separate affine, stable, no evidence of the bug.

## Why both softplus and soft_clip (complementary)

| Guard | Bounds | Mechanism |
|---|---|---|
| **softplus** | large-positive `a` (value + gradient) | scale grows *linearly* not exponentially; gradient `sigmoid(a+c) ∈ (0,1)` is bounded, vs `d/da exp(a)=exp(a)` unbounded (the backward-pass exploder) |
| **soft_clip** | both tails of `a` and all of `b` | hard `tanh` bound `a,b ∈ [−c, c]`; this is what caps the large-*negative* `a` direction (`scale→0 ⇒ z=(x−b)/scale→∞`), which softplus alone does not |
| **fp32 cast** | precision-driven overflow | affine computed in float32 |

With `soft_clip=4`: scale ∈ ~`[0.03, 4.55]`, `b ∈ [−4, 4]`, gradients bounded. STARFlow
ships both together for exactly this reason.

## API

Threaded `make_tarflow(...)` → `MetaBlock.__init__(...)`:

```
use_softplus: bool = True     # softplus scale parametrization (else legacy exp)
soft_clip:    float = 4.0      # tanh bound on proj_out output; 0 disables (>0 convention)
```

fp32 affine is always on (no flag; no-op at today's f32 default, future-proofs bf16).
Module constant: `INV_SOFTPLUS_1 = 0.541324854612918` (so `softplus(0 + INV_SOFTPLUS_1) = 1.0`
exactly ⇒ identity at zero-init).

## Core change: single `_affine` helper (one source of truth)

```python
def _affine(self, a):
    """Raw log-scale a -> (scale, inv_scale, log_scale), in fp32.
    scale plays the role of exp(a) ('1/sigma'): inverse multiplies by inv_scale,
    forward multiplies by scale, logdet uses log_scale."""
    a = a.astype(jnp.float32)
    if self.use_softplus:
        s = jax.nn.softplus(a + INV_SOFTPLUS_1)    # 1.0 at a=0 -> identity init
        return s, 1.0 / s, jnp.log(s)
    return jnp.exp(a), jnp.exp(-a), a               # exp branch: byte-identical to v2 today
```

- `inverse`: `z = (xp - b) * inv_scale`; `logdet = -sum(log_scale, axis=(1,2))`
- `forward` scan body: `xi = zp[:, i] * scale[:, i] + b[:, i]`; final `logdet = +sum(log_scale)`

The `exp` branch returns `exp(-a)` directly, so under the default float32 working dtype
`use_softplus=False` reproduces the current code bit-for-bit (the `astype(float32)` is a
no-op there). The transformed output is cast back to the working dtype.

## soft_clip placement

In `_params`, immediately after `out = self.proj_out(h)` and before the split:

```python
if self.soft_clip > 0:
    out = self.soft_clip * jnp.tanh(out / self.soft_clip)
a, b = jnp.split(out, 2, axis=-1)
```

Bounds `a` and `b` together (matches STARFlow `get_proj_out`). `tanh(0)=0` ⇒ no effect at init.

## Identity-at-init invariant (keeps default-ON safe)

At zero-init `proj_out`: `a=b=0` ⇒ softplus scale `= softplus(0.5413) = 1.0`, `log_scale=0`,
soft_clip `tanh(0)=0` no-op ⇒ `z = x`, `logdet = 0`. The untrained flow is still the
identity, so existing "untrained ≈ base" tests hold and default-ON is behavior-safe at init.
(Trained behavior differs from exp, by design.)

## EMA / buffer seam

The new knobs are static config (graphdef, not `nnx.Param`/`Variable`/`Mask` state); `sos_embed`
is unchanged. The EMA averaging seam is therefore untouched. A test will re-assert the seam
(no non-Param state captured by `optax.ema`) since this area has been historically sensitive.

## Test plan

1. `softplus_identity_at_init` — untrained softplus flow `log_prob ≈ base_log_prob`.
2. `softplus_roundtrip` — `forward∘inverse ≈ id` (random weights, softplus on).
3. `softplus_logdet_matches_numerical` — block logdet vs numerical Jacobian determinant.
4. `soft_clip_bounds_params` — large `proj_out` weights ⇒ `|a|, |b| ≤ soft_clip`.
5. `exp_path_unchanged` — `use_softplus=False` byte-identical to a saved exp snapshot.
6. `stability_regression` — stress input/large weights: softplus stays finite where the exp
   path overflows. Locks in the diagnosis.
7. `make_tarflow` defaults ⇒ `use_softplus=True`, `soft_clip=4.0` on every block.
8. Audit existing `tests/normalizing_flows/transformer_flow/test_model.py` (esp. analytic
   Jacobian and untrained≈base): update any assertion that assumes the exp logdet formula
   on the now-softplus default.

## Out of scope (future, evidence-driven)

- Optional softplus for MAF's `AffineTransformer` (`bijections/transformers.py`) — opt-in,
  default OFF, only if MAF ever destabilizes. MAF has no SOS free-scale redundancy and is
  currently stable.
- Pipeline-level NaN/grad-skip recovery guard — only if a model still diverges despite the
  source-level guards.

## Files touched

- `src/gensbi/normalizing_flows/transformer_flow/blocks.py` — `INV_SOFTPLUS_1`, `_affine`,
  soft_clip in `_params`, `use_softplus`/`soft_clip` on `MetaBlock`, fp32 in inverse/forward.
- `src/gensbi/normalizing_flows/transformer_flow/model.py` — plumb `use_softplus`/`soft_clip`
  through `make_tarflow`.
- `tests/normalizing_flows/transformer_flow/test_model.py` (+ a new test module if cleaner).
