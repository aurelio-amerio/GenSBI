# PixelDiT port — design spec (2026-06-12)

## Goal

Port the PixelDiT architecture (arXiv:2511.20645, NVIDIA; reference code in
`reference/PixelDiT/`) into `src/gensbi/experimental/models/pixeldit/` as a
JAX/flax-nnx, **channel-last** conditional flow-matching model for pixel-space
fields. Train with GenSBI's existing **CondOT flow matching** and sample with
the existing **ODE pipeline** — neither their REPA objective nor their
FlowDPM-Solver sampler is ported.

Context: FieldDiT (the previous in-house hybrid) underperformed and is
stalled. PixelDiT is its successor candidate. Per the project rule, the port
is **faithful to the reference** (norms, activations, init, block structure);
unification/ablation against Flux1 blocks is deferred to promotion time.

## Reference summary (what we are porting)

PixelDiT is a VAE-free, dual-level DiT operating directly on pixels:

1. **Patch-level pathway** (semantics): aggressive `p×p` patchify (`p=16` in
   the paper) → linear embed to width `D` → `N` MMDiT blocks. Output
   `s_cond = silu(t_emb + s_N)`, one semantic token per patch.
2. **Pixel-level pathway** (texture): each pixel embedded to a tiny width
   `D_pix` (16 in the paper) + 2D sincos absolute positions, grouped per patch
   to `(B·L, p², D_pix)`. `M` **PiT blocks**, each:
   - **pixel-wise AdaLN**: `Linear(D → n_mod·D_pix·p²)` from the patch's
     semantic token → per-pixel modulation parameters;
   - **token compaction**: flatten the patch's pixels (`p²·D_pix`) → linear
     compress to `attn_dim` → global RoPE attention over the `L` patch
     positions → linear expand back to pixels;
   - per-pixel MLP (GELU).
3. **Final layer**: RMSNorm + zero-init linear per pixel → output field.

Training target (verified in `c2i/src/diffusion.py`): `x_t = t·x + (1−t)·ε`,
`v = x − ε` — identical to GenSBI's CondOT path convention (noise at `t=0`).
The model output **is a velocity**; our ODE sampler consumes it directly.

## Decisions (locked during brainstorming)

- **Faithful reimplementation** of block internals: RMSNorm (`eps=1e-6`),
  SwiGLU FFN (LLaMA-style, `2/3·4D` width, no biases), sequential attn→FFN,
  their fractional-position 2D RoPE, their `TimestepConditioner` with
  `max_period=10` (not Flux's 10 000). Reuse only exactly-equivalent
  primitives: `flux1.math.attention`/`apply_rope` (real-arithmetic rotation ≡
  their complex cis multiply), `recipes.utils` patchify/id helpers where they
  apply, `nnx.RMSNorm`.
- **Conditioning = MMDiT cond tokens** (t2i topology, their
  `MMDiTBlockT2I`): patch pathway jointly attends over obs patch tokens +
  cond tokens; the pixel pathway is conditioned **only** through `s_cond`,
  exactly as in the paper. No Flux-style single-stream stage (PixelDiT has
  none).
- **Configurable cond id embedding** (this is where we generalize beyond the
  reference, which hardcodes text): additive id embedding via the existing
  `FeatureEmbedder` kinds `{"absolute", "pos1d"}` (or `"none"`), plus an
  independent `use_cond_rope: bool` for the reference's separate 1D rope on
  cond q/k. Faithful default = `"absolute"` + `use_cond_rope=True` (their
  `y_pos_embedding` + `use_text_rope`). For unordered θ-vector conditioning,
  `"absolute"` + `use_cond_rope=False` is the sensible setting. Because each
  stream's rope is applied before the joint q/k concat, obs and cond
  positional treatments are independent — no shared id space needed.
- **Init follows the c2i recipe** (deliberate mix, approved): zero-init *all*
  adaLN modulation linears + final layer → output is exactly zero at init
  (matches the FieldDiT verified-alive test pattern). The t2i reference only
  zero-inits the final layer; reachable via a flag (`zero_init_blocks=False`).
- **Scope = architecture + pipeline gates**: the phase ends with the model
  training and sampling end-to-end on a toy field through
  `FieldConditionalPipeline`, with FieldDiT-style learning gates green.
- **Skipped from their recipe** (training-config refinements, not
  architecture): REPA/DINOv2 alignment loss, logit-normal t-sampling,
  timeshift, CFG/null-cond training (`conditioned=False` raises, as in
  FieldDiT), FlowDPM-Solver, EMA-decay/grad-clip schedule specifics.
- 2D fields only in this phase (the reference is 2D); 1D fields and
  spatially-structured cond (rope2d cond grids, Kontext-style) are future
  work.

## Package layout

`src/gensbi/experimental/models/pixeldit/`:

| File | Contents |
|---|---|
| `rope.py` | `precompute_freqs_cis_2d(head_dim, h, w, theta=10000, scale=16)` — fractional positions `linspace(0, scale, ·)`, x/y interleaved pair layout, emitted in the `(..., 2, 2)` rotation format `flux1.math.apply_rope` consumes; `precompute_freqs_cis_1d` for cond rope (integer positions, full head dim). |
| `modules.py` | `SwiGLU`, per-pixel `MLP` (GELU), `TimestepConditioner` (`max_period=10`), `FinalLayer` (RMSNorm + zero-init linear), 2D sincos abs-pos table builder. |
| `blocks.py` | `MMDiTBlock` (faithful `MMDiTBlockT2I`: per-stream qkv/proj/SwiGLU, RMSNorm pre-norms, 6-param adaLN per stream from `c = silu(t_emb)`, joint attention, per-stream rope) and `PiTBlock` (pixel-wise AdaLN incl. both `pit_post_modulation` variants, compress→global-attend→expand, per-pixel MLP). |
| `embedders.py` | `PatchTokenEmbedder` (linear on flattened patch), `PixelTokenEmbedder` (per-pixel linear + 2D sincos abs-pos, group to `(B·L, p², D_pix)`), `CondTokenEmbedder` (linear `D_c→D` + RMSNorm + configurable `FeatureEmbedder` id embedding). |
| `model.py` | `PixelDiTParams` (dataclass, `FieldDiTParams` style) + `PixelDiT`. |

Exported as `from gensbi.experimental.models import PixelDiT, PixelDiTParams`.

## Model contract

```
PixelDiT(params)(t, obs, cond, obs_ids=None, cond_ids=None,
                 conditioned=True, guidance=None) -> velocity
```

- `obs`: `(B, H, W, C)` channel-last; `cond`: `(B, K, D_c)` tokens (a θ-vector
  `(B, k, 1)` from the existing pipeline contract is K tokens with `D_c=1`).
- `obs_ids`/`cond_ids` accepted-and-ignored (tables built internally for the
  fixed `field_shape`), so `FieldConditionalWrapper` and
  `FieldConditionalPipeline` work without modification.
- Output: `(B, H, W, C)` velocity. Exactly zero at init (default init).
- All positional tables (2D rope, cond rope, pixel sincos) precomputed at init
  and stored in `RopeIds`-style `nnx.Variable` buffers — no dict caches, no
  dynamic shapes, JIT-friendly.

Channel-last replaces the reference's `unfold`/`fold` with plain reshapes:
`(B,H,W,C) → (B, Hs, p, Ws, p, C) → (B, L, p²·C)` for patches, and the
analogous grouping for pixel tokens.

`PixelDiTParams` knobs (defaults to be finalized in the plan): `in_channels`,
`field_shape`, `patch_size`, `hidden_size D`, `pixel_hidden_size D_pix`,
`patch_depth N`, `pixel_depth M`, `num_heads`, `pixel_attn_hidden_size`,
`pixel_num_heads`, `cond_dim K`, `cond_in_channels D_c`,
`cond_id_embedding ∈ {"absolute","pos1d","none"}`, `use_cond_rope`,
`use_pixel_abs_pos`, `pit_post_modulation`, `zero_init_blocks`,
`rope_scale=16.0`, `theta=10_000`, `param_dtype=bf16`. The paper-B-scale
config (`D=768, N=12, M=2, D_pix=16, p=16`) is exercised by the opt-in gate-3
smoke test rather than shipped as a named preset.

## Training & sampling wiring

- `FieldConditionalPipeline` with the CondOT flow-matching `GenerativeMethod`
  — no pipeline changes expected. Data contract unchanged:
  `(obs (B,H,W,C), cond (B,K,D_c))`.
- Sampling through the existing ODE solver via `FieldConditionalWrapper`.
- Known inherited limitation: field `log_prob` is broken upstream
  (divergence path assumes token-shaped events) — out of scope here, same as
  for FieldDiT.

## Verification

1. **Unit level**: shapes, finiteness, exact-zero-at-init, dtype policy
   (timestep sinusoid in f32 before cast — FieldDiT lesson), both
   `pit_adaln_post_modulation` variants, cond id-embedding modes.
2. **Gate 1 — gradient aliveness**: after optimizer step(s), every subtree
   (patch embedder, cond embedder, MMDiT blocks, PiT blocks, final layer,
   time conditioner) has nonzero grads/updates. FieldDiT lesson: subtrees that
   reach the output only through zero-init modulation need **two** steps.
3. **Gate 2 — tiny overfit + cond sensitivity**: loss drops by orders of
   magnitude on a tiny dataset; structurally different conds give different
   outputs (uniform-shift probes are annihilated by the RMSNorm in the cond
   embedder — use structurally different conds).
4. **Gate 3 — opt-in realistic smoke**: paper-B-ish config at a realistic
   field size, params/memory/walltime printout.
5. **Parity script** (`scripts/` or `tests/` opt-in, run manually, not CI):
   tiny config, copy weights from the PyTorch reference, compare forward
   outputs to tolerance. This is the only real proof of "faithful". If a
   torch-capable env is unavailable, drop and rely on line-level code review
   against the reference.

## Deviations (post-implementation)

Intentional deviations from the reference discovered or confirmed during implementation and the parity script:

1. **`t` shape normalisation at model boundary.** `FieldConditionalWrapper._expand_time` passes `t` as `(B, 1)`, while the faithful `_timestep_embedding` (which mirrors the reference) expects a 1-D `(B,)` vector. `PixelDiT.__call__` calls `t.ravel()` at the entry point so the internal port stays untouched. No change to the pipeline or wrapper.

2. **Gate-2 overfit uses 3 000 steps, not ~300.** CFM loss has high stochasticity from the random `t` draw; convergence by 2 orders of magnitude requires more steps on a tiny fixed dataset than the plan originally estimated. The actual step count is documented directly in the test.

3. **Patch-token flatten is pixel-major (`(i·p+j)·C + c`), not channel-major.** The reference uses `F.unfold` (NCHW → channel-major), whereas the channel-last reshape `(B, Hs, p, Ws, p, C) → transpose(0,1,3,2,4,5) → (B, L, p²·C)` produces pixel-major ordering. The two orderings are internally consistent (patch embed, pixel grouping, and `unpatchify` all use the same convention), but loading reference checkpoint weights into `s_embedder` requires a kernel-row permutation. The parity script (`scripts/pixeldit_parity.py`) applies this permutation; its header explains the mapping.

4. **Torch parity verified at max |Δ| ≈ 3 × 10⁻⁸ (float32).** No numerical deviations beyond floating-point rounding.

## Out of scope / future work

CFG & null-cond training; 1D fields; spatially-structured (2D-grid) cond
tokens; logit-normal t / timeshift knobs; REPA; FlowDPM-Solver; field
`log_prob` fix; any merge/ablation with Flux1 blocks (promotion-time work).
