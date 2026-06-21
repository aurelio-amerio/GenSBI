# Design Spec — FieldDiT: conditional flow-matching for field-level inference

**Date:** 2026-06-09
**Status:** Draft for review/attack. Positions are taken deliberately so they can be contested.
**Authors:** Aurelio Amerio + Claude (brainstorming session)
**Supersedes:** `2026-06-09-pixel-space-hybrid-sbi-model-design.md` (same effort, reframed as FieldDiT).

---

## 1. Goal & scope

**FieldDiT** is a conditional flow-matching model that generates a **field** (1D or 2D) directly in **pixel/signal space**, given a conditioning input. It is one model with two faces:

- **Emulator** — conditioned on a few **θ-parameters**, it samples `p(field | θ)` (e.g. physics/cosmology params → field realization). Here θ describes the *whole* field; the condition is **global/statistical**.
- **Rich-conditioned generator** — conditioned on a structured input (a correlated 1D feature vector; later, an image), it samples `p(field | condition)`.

The unifying statement: **the condition can be anything, embedded into tokens by a per-modality embedding network**; joint attention in the transformer core is the single conditioning mechanism (§2). A handful of scalars → a `Linear` producing a few tokens; a sequence/vector → a 1D encoder; (Phase 2) an image → co-tokenization with the obs.

It unifies two existing designs:

- **Flux1 (MMDiT)** — double-stream + single-stream transformer; the mechanism for **rich conditioning** (merges obs and cond token streams via joint attention). Already supports 1D/2D/unstructured data via `id_embedding_strategy`, including **semantic ids** (the Kontext alignment primitive — see §10).
- **ResUViT (SiD2-style)** — conv encoder/decoder with global (time) modulation, residual skip connections, and a transformer at the bottleneck; the *sample-efficient, multiscale* mechanism for **field/image data** in pixel space, with **no VAE**.

FieldDiT = the ResUViT skeleton with its bottleneck **upgraded from self-attention to the Flux1 MMDiT**, so conditioning enters as a token stream instead of by spatially-aligned channel concatenation.

### Dimensionality
Interfaces are **dimension-agnostic** (`*_2d` / `*_1d` variants). **2D is implemented first**; 1D is a parallel path that reuses the same machinery (GenSBI already ships `init_ids_1d`, `AutoEncoder1D`, `Embedded1DModel`).

### Phasing (scope discipline)
- **Phase 1 (this spec's build target):** the general **embed-any-condition-to-tokens** path, **anchored and validated on global/statistical conditioning** — the θ-emulator (GRF 256² as the toy/test) and non-image rich conditions (e.g. a correlated 1D feature vector). flagged-C ON (§2, R2). This is a baseline that is *correct* for global/statistical conditions and *usable* for others.
- **Phase 2 (deferred, §10):** **image / spatially-aligned conditioning** via **Kontext-style co-tokenization** (concat obs+cond tokens on a shared grid, separated by RoPE semantic id). Delicate (shared tokenizer, read-only cond, CFG over images) and **purely additive** — it does not change the Phase 1 architecture.

### Non-goals
- **Latent diffusion / VAE-based** generation of the *obs/field*. We keep the obs codec deterministic and in pixel space (§3.1; cost in R4). (A *conditioning* embedder may itself be a VAE if a pretrained one is convenient — separate from the obs path.)
- A monolithic mega-model. We specify **component interfaces** so each task instantiates only the pieces it needs (§5).
- Posterior inference where the obs is low-dim (a few θ scalars) and the cond is the rich object — **already served by stock Flux1** (the lensing case). FieldDiT is for the *field-output* regime.

---

## 2. Guiding principle (the invariant)

**The conditioning mechanism must match the structure of the conditioning information.** FieldDiT resolves "structure" by **embedding everything into a token stream** and letting joint attention do the work, with a global modulation pathway for time and (by default) a condition summary:

| Information | Mechanism | Where it enters |
|---|---|---|
| **Time** `t` (the integration variable) | **modulation** (scale/shift/gate) | every conv resblock *and* every transformer block |
| **Condition** (θ scalars, vector; Phase 2: image) | **tokens + joint attention** | the MMDiT core |
| **Condition's global summary** (flagged-C, **default ON**) | **modulation** | added to the modulation vector → modulates the **decoder** resblocks (R2) |

Why the conv codec stays **condition-token-agnostic**: a conv resblock can only inject a *global* vector (a scale/shift), not a structured token stream. The reference `ResidualBlock2D` says so explicitly — *"spatial conditioning maps … are intentionally not handled here; the backbone wrapper should decide where/how often to inject spatial conditions."* The reference's wrapper chooses **early-fusion channel-concat**, which **requires spatial alignment**. FieldDiT routes the condition into the **MMDiT token stream**, removing that constraint; spatially-aligned alignment is handled separately and later by co-tokenization (§10).

### Unified modulation (the resolved FiLM-vs-AdaLN question)
Both the conv codec and the transformer core are modulated by the **same mechanism**: a single shared predictor consumes
`vec = time_emb ⊕ [pooled_cond_summary] ⊕ [guidance]`
and emits **(scale, shift, gate)**, applied as `out = residual + gate · (modulated_path)` with the **gate zero-initialized** (identity at init). The only difference between the two places it is applied:

- **Conv codec → AdaGN-zero**: modulation over **GroupNorm** (norm's own affine off; predicted scale/shift is the sole affine). GroupNorm is the proven normalization for conv image backbones.
- **Transformer core → AdaLN-zero**: modulation over **LayerNorm**, native to tokens (this is exactly Flux1's `Modulation`).

This is "FiLM + a conditioning-predicted zero-init gate," unified into one predictor (R7 reuse, R5 isolation). The gate adds (a) identity-at-init and (b) conditioning-dependent block strength.

> **Flagged for later (near R2 / validation):** for a *field/SBI emulator*, the normalization choice is not cosmetic — any norm that pools over space (GroupNorm) removes per-field spatial mean/variance, part of what we may be trying to emulate (e.g. power-spectrum structure). Deserves a dedicated study; **not** decided here.

---

## 3. Architecture

```
   condition (θ scalars | sequence | vector ;  Phase 2: image)
                     │
              ┌──────▼───────┐
              │ CondEmbedder │  Linear (few params) | 1D-enc (sequence/vector)   [Phase 2: image → co-tokenize]
              └──────┬───────┘
                     │ cond tokens (+ pooled summary → vec, default ON)
                     │
 obs/field x_t (full res, NHWC)                         time t ─┐
        │                                                       │
   ┌────▼──────────┐   pos_skip (per resolution, +)             │ vec = t ⊕ cond_summary ⊕ [guid]
   │  ObsEncoder    │───────────────────────────────┐          │
   │  (conv ↓, AdaGN│  neg_skip (per resolution, −) │          │   (encoder: time-only modulation)
   │   time-only)   │──────────────────┐            │          │
   └────┬──────────┘                   │            │          │
        │ feature map @ meeting grid    │            │         ▼
   ┌────▼─────┐ patchify_2d(size p)     │            │   ┌──────────────────────────────┐
   │ tokenize  │ → obs tokens            │           └──►│   MMDiT core (Flux1)          │
   └────┬─────┘                         │            ┌───│  DoubleStream(obs,cond)×N     │
        ▼                                │            │   │  → concat → SingleStream×M    │
   obs tokens ───────────────────────────────────────────│  joint attention, rope ids    │
                                         │            │   │  AdaLN-zero(vec)              │
   ┌────▼─────┐ depatchify_2d            │            │   └──────────────────────────────┘
   │ untokenize│ ◄──────────────────────────────────-┘
   └────┬─────┘  refined feature @ grid  │
        │                                │
   ┌────▼──────────┐  − neg_skip, then + pos_skip (SiD2 residual skips, in conv space)
   │  ObsDecoder    │◄───────────────────┘
   │  (conv ↑, AdaGN│   (decoder: time ⊕ cond_summary modulation — flagged-C, R2)
   │   + cond_summ) │
   └────┬──────────┘
        ▼
   velocity field, full res  (pixel-space flow-matching target)
```

### 3.1 Why pixel space (not a VAE on the obs)
SBI is judged by **statistical fidelity and calibration**, not perceptual quality. A VAE — especially with perceptual/adversarial losses — hallucinates plausible detail and imposes a lossy, possibly *miscalibrated* noise floor on the inference. We keep the obs codec **deterministic and end-to-end** (no sampling bottleneck, no KL, no perceptual/GAN loss) so the only objective shaping it is the flow-matching loss. Price: the transformer runs at the conv+patchify-reduced resolution (R4).

### 3.2 The load-bearing knob: at what resolution do obs and cond meet?
The meeting-grid resolution (= number of obs tokens entering the MMDiT) is the **primary tunable**, set by two configurable hyperparameters (R1, R4):

```
tokens = (H / (2^D · p)) · (W / (2^D · p))
   D = ObsEncoder depth (number of stride-2 stages)
   p = patch size for patchify_2d after the conv encoder (1 = no patchify)
hidden_size = transformer feature width — an INDEPENDENT knob
```

The knob is *free*, but its **safe range depends on the conditioning regime** — physics, not a default:

- **Global / statistical condition** (θ-emulator; GRF's α; correlated vectors): the condition controls *statistics*, not *locations*. A coarse meeting grid is fine; fine-scale control comes from flagged-C (R2). → reduce aggressively.
- **Spatially-aligned condition** (Phase 2: super-res, inpainting): the condition must place detail *at specific pixels*. This needs both a **fine enough grid** *and* the **co-tokenization mechanism** (§10) — a global summary cannot localize. → keep the grid fine; handle in Phase 2.

**Defaults (Phase 1, all overridable):** the *knobs* carry defaults — `D = 3`, `p = 2`, `hidden_size ≈ 768` — but the **token count is derived from the field shape at init time, not prescribed.** There is no meaningful fixed token default: `tokens = (H/(2^D·p))·(W/(2^D·p))`, so it depends on the data. With these defaults a 256×256 field lands at 16×16 = 256 tokens (a 128² field → 64, etc.). The intended workflow: instantiate with `D=3, p=2`, **inspect the resulting token count** (known once the data shape is known at init), and adjust `D`/`p` if it is too large or too small. (`p = 4` quarters the token count for a leaner model.)

### 3.3 Components, concretely (256×256×1 obs, default config)
- **ObsEncoder** (`conv ↓`): port the SiD2 stage structure onto GenSBI's JAX conv blocks — `ResnetBlock2D` + `Downsample2D` (`autoencoder_2d.py`), used **deterministically** (no `DiagonalGaussian`). 256→128→64→32 (D=3), channels per `encoder_widths`. Each resblock modulated by **time only** via AdaGN-zero (encoder stays condition-free for modularity). Captures `pos_skip` (pre-downsample) and `neg_skip` (post-downsample) per stage.
- **Tokenizer**: `patchify_2d(size=p)` on `(B, h, w, C)` → `(B, (h/p·w/p), C·p²)`, then `Linear` to `hidden_size`. `init_ids_2d(dim=(h,w), size=p, semantic_id=obs)` builds the matching rope2d ids.
- **CondEmbedder (pluggable, per modality)** → cond tokens at `hidden_size` (+ pooled summary):
  - few θ scalars → `nnx.Linear` → a few tokens; `id_embedding_strategy = "absolute"` (order-free).
  - sequence / correlated vector → 1D encoder or `Linear`; `pos1d`/`rope1d`.
  - *(Phase 2)* image → co-tokenized with the obs (§10).
- **MMDiT core**: Flux1 `DoubleStreamBlock` × `depth` (separate obs/cond modulation, joint attention) → concat → `SingleStreamBlock` × `depth_single_blocks`. `vec = time_emb ⊕ pooled_cond_summary ⊕ [guidance]`. obs tokens carry rope2d ids; cond tokens their modality-appropriate ids. (Reuse `Modulation`, `EmbedND`, `timestep_embedding`.)
- **Untokenizer**: `Linear` back to `C·p²` → `depatchify_2d(size=p)` → `(B, h, w, C)`.
- **ObsDecoder** (`conv ↑`): mirror of ObsEncoder (`Upsample2D` + `ResnetBlock2D`), 32→64→128→256 → velocity field. AdaGN-zero modulation by **time ⊕ cond_summary** (flagged-C, R2). **SiD2 residual skips applied here, in conv space, outside the patchify↔MMDiT sandwich** (§3.4). Final conv zero-init (`LastLayer`-style).
- **Modulation predictor**: one shared `Modulation`-style module feeding both codec (over GroupNorm) and core (over LayerNorm) — see §2.

### 3.4 Skip connections (R3 = on, SiD2 residual scheme)
Skips are **on by default**, **SiD2 residual scheme** (add/subtract, *not* concat → no channel growth), as in `ResidualUViT.decode`:
```
# per resolution, decoder side, in conv-feature space:
x = x − neg_skip        # subtract matching post-downsample feature
x = upsample(x)
x = x + pos_skip        # add matching pre-downsample (high-res) feature
```
Makes the U-Net a **stable residual around "pass the encoder feature straight through,"** pairing with the zero-init gate. The arithmetic is **in conv space** — the decoder un-patchifies to the conv grid/width *before* `x − neg_skip`, so **skips bypass the transformer**. Clean boundary; no conflict with patchify.

**Honest scope of skips (links to R2):** skips carry **obs** detail around the MMDiT; they do **not** propagate the *condition* to fine scales. flagged-C does that (R2).

### 3.5 Config object
Mirror `Flux1Params`: a `@dataclass FieldDiTParams` holding (representative):
`in_channels, field_shape, ndim, encoder_widths (→ depth = len), res_blocks_down/up, patch_size, hidden_size, num_heads, mlp_ratio, depth, depth_single_blocks, qkv_bias, id_embedding_strategy=(obs,cond), theta, axes_dim/id_merge_mode, cond_spec (modality), context_in_dim, use_cond_summary_in_vec=True (flagged-C, decoder), use_cond_token_stream=True, use_skips=True, use_gate=True, norm_groups, guidance_embed, param_dtype, rngs`.

---

## 4. Data flow (flow-matching, per ODE step)
1. Noise added to the field in **pixel space**; target velocity is in pixel space.
2. `cond_tokens, cond_summary = CondEmbedder(cond)`; `vec = time(t) ⊕ cond_summary ⊕ [guidance]`.
3. `feat, pos_skips, neg_skips = ObsEncoder(x_t, time(t))`  *(conv ↓, AdaGN-zero, time-only)*.
4. `obs_tokens = Tokenize(feat)`  *(patchify + Linear)*.
5. `obs_tokens = MMDiT(obs_tokens, cond_tokens, vec)`  *(joint attention, AdaLN-zero)*.
6. `feat = Untokenize(obs_tokens)`  *(Linear + depatchify)*.
7. `v = ObsDecoder(feat, vec, pos_skips, neg_skips)` → full-res velocity  *(decoder AdaGN-zero by time ⊕ cond_summary)*.
8. Loss = flow-matching MSE in pixel space. **Encoder, transformer, decoder are one network trained end-to-end** — no separate codec objective. Runs **every ODE step** (pixel-space, not latent — R4).

---

## 5. Modular interfaces (design for isolation — R5)
Small self-contained, independently testable layers:

- `ObsEncoder(x, time_emb) -> feat, pos_skips, neg_skips`
- `Tokenize(feat) -> obs_tokens, obs_ids` / `Untokenize(obs_tokens) -> feat`
- `CondEmbedder(cond) -> cond_tokens, cond_ids, summary` — scalar / sequence variants (Phase 2: image)
- `MMDiTCore(obs_tokens, cond_tokens, vec, ids) -> obs_tokens`
- `ObsDecoder(feat, vec, pos_skips, neg_skips) -> field`
- `Modulation(vec) -> (scale, shift, gate)` — shared; over GroupNorm (codec) or LayerNorm (core)

**Instantiations (same components, different wiring):**
- **θ-emulator / GRF toy (Phase 1):** CondEmbedder = `Linear` on few scalars → a few `absolute` tokens; flagged-C ON; aggressive token reduction (D high, p∈{2,4}).
- **Rich non-image cond (Phase 1):** CondEmbedder = 1D encoder → token stream; double-stream active; flagged-C ON.
- **Spatially-aligned image2image (Phase 2, §10):** Kontext-style co-tokenization; optionally share obs/cond tokenizer weights (R6); keep the meeting grid fine.
- **Lensing (posterior, low-dim obs):** out of scope — stock Flux1.

---

## 6. Validation plan
Diagnostics must live in the **right space**. For field emulation the target is `p(field | condition)`:
- **Power-spectrum recovery**: generated vs simulator fields across the prior (the GRF's sufficient statistic) — first quantitative check on the toy task; directly tests whether flagged-C's per-band modulation reproduces the spectral slope (R2).
- **Calibration in field/statistic space** (SBC/TARP on field summaries), **not** θ-space. The stale θ-space SBC/TARP block in `grf.py` must be **rewritten or deleted**.
- **Reconstruction sanity**: deterministic codec round-trip error as an upper bound on achievable fidelity.
- **Build order**: validate the emulator path on the GRF toy (power spectrum + field-space calibration) before any Phase 2 work.

---

## 7. Risks / Open Decisions (the review surface — attack these)

**R1 — Meeting-grid resolution.** *Resolved into a configurable knob* (`D`, `p`; §3.2), `hidden_size` independent. *Residual risk:* safe range is **regime-dependent** — coarse is fine for global/statistical cond (Phase 1), too coarse breaks spatially-aligned cond (Phase 2 / §10). Knob defaults `D=3`, `p=2`, `hidden_size≈768`; the **token count is derived from the field shape at init, not a fixed default** (§3.2).

**R2 — Conditioning reaching fine scales.** *Resolved* — and it is **the fine-scale face of R1**, not a separate risk. The condition enters only at the coarse meeting grid; can it shape fine output scales?
- **Global/statistical regime (Phase 1):** YES, via **flagged-C (default ON, decoder)**. Each decoder stage ≈ a spatial-frequency band, and each block has its own `vec→(scale,shift,gate)` projection, so **per-stage modulation = per-band amplitude control**; a global scalar (e.g. GRF's α) → per-band amplitudes → spectral slope. This is the load-bearing justification for "coarse grid is OK for the emulator." It is the **same pattern as latent diffusion** (condition at coarse grid, synthesize fine in decode) — and FieldDiT is better positioned: its decoder is itself conditioned and trained end-to-end, and skips carry the fine *obs* detail an LDM must encode.
- **Spatially-aligned regime (Phase 2):** a global summary **cannot localize** by construction; needs the R1 grid + co-tokenization (§10). Deferred.
- *Encoder stays time-only* (modularity); decoder-only flagged-C suffices because post-skip resblocks re-weight the combined feature per condition.

**R3 — Skip connections.** *Resolved: ON by default, SiD2 residual scheme* (§3.4). *Residual risk:* residual vs concat skips may under/over-smooth on some data; concat is a fallback.

**R4 — Pixel-space resolution cap.** No VAE on obs caps the interaction resolution (conv+patchify grid). 256×256 default, **tunable** via `D`, `p`, `hidden_size`. *Attackable:* for very large fields a **statistics-preserving** (not perceptual) VAE-on-obs may become the lesser evil. Name the breakpoint during review.

**R5 — Monolith creep.** *Mitigated* by §5 interfaces as **small self-contained layers**; tasks are instantiations, not forks. Phasing (§1) is the main scope guard.

**R6 — Cond/obs codec sharing.** Relevant only for **img2img** (Phase 2) — enables Kontext co-tokenization (obs and cond share a tokenizer so their grids/ids align). **Not a Phase 1 concern.**

**R7 — Reuse vs rebuild.** The ResUViT reference is **Keras (bayesflow)**; GenSBI is **JAX/flax.nnx** — *the reference code cannot be imported.* "Reuse maximally" = (a) reuse GenSBI's JAX layers: `DoubleStreamBlock`/`SingleStreamBlock`/`Modulation`/`EmbedND`/`LastLayer` (flux1), `ResnetBlock2D`/`Down/Upsample2D` (autoencoder_2d, stripped of `DiagonalGaussian`), `patchify_2d`/`depatchify_2d`/`init_ids_*` (recipes/utils); and (b) **port the SiD2 patterns**: residual skips, modulation-everywhere, zero-init last conv, transformer-at-bottleneck. *Risk:* conv blocks may carry VAE assumptions (GroupNorm groups, unused `hs` list) — audit before reuse.

---

## 8. Relationship to existing code
- `src/gensbi/models/flux1/layers.py` — `DoubleStreamBlock`, `SingleStreamBlock`, `Modulation` (AdaLN-zero), `EmbedND`, `LastLayer`, `timestep_embedding`: MMDiT core + shared modulation.
- `src/gensbi/models/flux1/model.py` — `Flux1Params` + `Flux1` assembly: the pattern to mirror for `FieldDiTParams` + `FieldDiT`. Also the **semantic-id RoPE** mechanism reused for Phase 2 co-tokenization (§10).
- `src/gensbi/experimental/models/autoencoders/autoencoder_2d.py` — `ResnetBlock2D`, `Downsample2D`, `Upsample2D`, `Encoder2D`/`Decoder2D`: the Obs codec, **stripped of `DiagonalGaussian`**, retrofitted with SiD2 residual skips + AdaGN-zero. (1D variants exist for the 1D path.)
- `src/gensbi/recipes/utils.py` — `patchify_2d`/`depatchify_2d`, `init_ids_2d`/`init_ids_1d`/`init_ids_joint`, `_resolve_embedding_ids`: tokenize boundary + id construction (incl. semantic ids).
- `src/gensbi/experimental/models/glue/embedder.py` — `Embedded2DModel`/`Embedded1DModel`: the conditioning-embedder precedent to generalize into the pluggable `CondEmbedder`.
- **Reference only (not importable):** `reference/bayesflow/networks/subnets/unet/` — SiD2 Residual U-ViT in Keras; source of the skip/modulation/bottleneck patterns.

---

## 9. Open questions for the author
1. **Normalization vs field statistics** (§2 flag): does GroupNorm's spatial pooling harm emulation of spatial-statistic targets (power spectrum)? Worth a small ablation before scaling up.
2. **1D path priority:** build the 1D variant alongside 2D, or strictly after the 2D path is validated on the GRF toy?
3. **Largest field resolution** beyond 256×256 that must remain viable in pixel space (sets the R4 breakpoint where a statistics-preserving VAE-on-obs is reconsidered).
4. **Phase 2 trigger:** what concrete image-conditioned task motivates building §10, and does it need pixel-aligned (super-res/inpainting) or merely same-grid conditioning?

---

## 10. Phase 2 (deferred): image / spatially-aligned conditioning

Not built in Phase 1; recorded so Phase 1 interfaces don't preclude it. **Additive** — it adds an image `CondEmbedder` and a co-tokenization path; it does not change the Phase 1 conv-codec + MMDiT + flagged-C.

**Mechanism (Kontext-style):** tokenize the conditioning image **on the same grid** as the obs, **concatenate** obs and cond tokens into one sequence, and separate them by the **semantic id of the RoPE positional encoding** — so joint attention sees aligned 2D positions while knowing which token is obs vs cond. GenSBI already has the primitives: `init_ids_2d(semantic_id=...)` and the obs/cond split in `init_ids_joint`.

**Open design challenges (the "delicate" parts):**
- **Shared tokenizer:** the cond image should pass through the *same* conv tokenizer as the obs (weight-shared, R6) so grids/ids align — couples the two codecs.
- **Read-only condition:** the cond image has no decoder and no skips → the U-shape becomes asymmetric.
- **Classifier-free guidance** over an *image* condition (drop/keep mask) is its own design.
- **Alignment validity:** only meaningful when obs and cond share resolution/grid; otherwise fall back to the Phase 1 non-aligned token-stream path.
