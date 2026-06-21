# FieldDiT Phase 1 — Handoff & Pre-Phase-2 Checklist

**Date:** 2026-06-09
**Branch:** `FieldDiT` (un-merged; base `main`)
**Plan executed:** `docs/superpowers/plans/2026-06-09-fielddit-phase1-model.md`
**Design spec:** `docs/superpowers/specs/2026-06-09-fielddit-design.md`

This document has two parts: **Part A** recaps what was built and how (so you know what to review), and **Part B** is the prioritized list of things to consider/address before starting Phase 2.

---

## Part A — Recap (what to review)

### What Phase 1 delivered

The `FieldDiT` model and its `FieldDiTParams` config: a conditional flow-matching network mapping `(t, noisy_field, cond) → velocity field` (same shape as the field), built as small, independently-tested `flax.nnx` modules. The model is callable, finite, **exactly zero at init**, and differentiable. It is **not** yet wired into a training pipeline (see Part B).

### Module map — `src/gensbi/experimental/models/fielddit/`

| File | Public symbols | Responsibility |
|---|---|---|
| `blocks.py` | `_safe_groups`, `ConvModulation`, `ModulatedResBlock2D` | AdaGN-zero modulated conv primitive (FiLM over GroupNorm + zero-init gate → identity at init) |
| `codec.py` | `Downsample2D`, `Upsample2D`, `ObsEncoder`, `ObsDecoder`, `Tokenizer`, `Untokenizer` | Conv U-Net halves (time-only-modulated encoder ↓ capturing SiD2 skips; time+cond-modulated decoder ↑ with subtract/add skips and zero-init output conv) + the patchify boundary |
| `cond.py` | `ScalarCondEmbedder` | Phase-1 condition embedder: condition → token stream + pooled summary |
| `core.py` | `MMDiTCore` | Flux1 joint-attention bottleneck (rope2d obs ids + absolute cond ids; cond concatenated before obs) |
| `model.py` | `FieldDiTParams`, `FieldDiT` | Config dataclass (derives hidden_size, depth, meeting grid, token count, ids) + end-to-end assembly/forward |
| `__init__.py` | re-exports `FieldDiT`, `FieldDiTParams` | also re-exported from `gensbi.experimental.models` |

Plus a fix to a shared util: `src/gensbi/recipes/utils.py::depatchify_2d` gained a `grid=(h, w)` argument so it actually inverts `patchify_2d` (it was broken before — couldn't infer the grid from token count — and had **zero callers**, so the change is backward-compatible).

### How it was built

Subagent-driven development: 10 plan tasks, each executed by a **fresh implementer subagent**, then gated by **two reviews — spec-compliance first, code-quality second** — with all findings resolved before moving on. Mechanical transcription tasks used a fast model; the two integration-risky tasks (`MMDiTCore`, the `FieldDiT` assembly) and the whole-feature review used the most capable model. Before the transformer task, the real Flux1 `DoubleStreamBlock`/`SingleStreamBlock`/`EmbedND`/`FeatureEmbedder` call conventions were verified against source, which is why the bottleneck wiring landed first-try.

### Commit map (22 commits, `0772ce8..HEAD`)

| Task | feat/fix commit | follow-up (review fixes) |
|---|---|---|
| 1 blocks | `56b31c5` | `00ac246` (docstring) |
| 2 down/up | `6e40662` | `2c729b9` (annotations + finiteness asserts) |
| 3 ObsEncoder | `adb735f` | `4963b3c` (annotate `widths`; finiteness) |
| 4 ObsDecoder | `3716075` | `202fa6f` (consolidate test imports) |
| 5 depatchify + Tokenizer | `d9b79ad`, `75d0626` | `daae176` (error-path + stronger shape test) |
| 6 ScalarCondEmbedder | `4b7ce21` | `384eec0` (2D-input docstring caveat; finiteness) |
| 7 MMDiTCore | `c37fedb` | `6b49ac8` (prebatched-ids + bf16 tests) |
| 8 FieldDiTParams | `1519d6b` | `0735603` (patch_size validation test; document `vec_in_dim`) |
| 9 FieldDiT assembly | `7e69653` | `104c887` (bf16 forward + guidance-hook tests) |
| 10 exports | `4e61b70` | `1f5fc6c` (relative import; newline; docstring) |
| — whole-feature follow-up | — | `4bb6108` (cond-token-count guard; annotate MMDiTCore ctor; fix deprecated `.value`) |

### Deviations from the literal plan (so you can spot them in review)

The plan dictated exact code; implementers transcribed it. The following **additive** changes were made on top (none altered a public name or signature):
- **Type annotations** added to constructor params in `codec.py`, `cond.py`, `core.py` to match the repo's annotated style (the plan's `ModulatedResBlock2D` was annotated; the helper classes weren't — now consistent).
- **Extra tests** added beyond the plan's: finiteness asserts; depatchify error-path (`ValueError`); a discriminating tokenizer shape test (`hidden ≠ token-count`); MMDiTCore prebatched-ids + bf16-default; `FieldDiT` bf16-forward + guidance-embed path + guidance-requires-guidance; `patch_size` validation; cond-token-count mismatch.
- **One behavioral guard** (`4bb6108`): `FieldDiT.__call__` now asserts `cond_tokens.shape[1] == cond_dim`. Without it, `cond_dim == 1` (single θ scalar — an intended config) would silently broadcast a wrong-but-finite output. **This is the one commit no independent reviewer saw before it landed** — worth a focused look, though it's a one-line static-shape assert.
- Minor hygiene: docstring clarity, import consolidation, trailing newline, replaced a deprecated `.value` with `[...]`.

### What to review, concretely

1. **`model.py` `FieldDiT.__call__`** — the modulation split is the load-bearing design choice: encoder gets `time_vec` (time only); decoder + core get `vec = time (+cond summary if `use_cond_summary_in_vec`) (+guidance)`. Confirm this matches your intent.
2. **`core.py` `MMDiTCore`** — the obs-rope / cond-absolute id handling and the cond-before-obs concat order. Mirrors `src/gensbi/models/flux1/model.py:352-460`.
3. **`codec.py` `ObsEncoder`/`ObsDecoder`** — the SiD2 subtract/add skip scheme and the zero-init `conv_out`.
4. **`recipes/utils.py` `depatchify_2d`** — the new `grid` arg and square-inference fallback.
5. **The test suite** under `tests/experimental/models/fielddit/` — 30 tests; check whether the asserted *nature* (shape/finite/zero-init/differentiable) is what you want pinned.

### Test status
- `tests/experimental/models/fielddit/` → **30 passed**
- `tests/experimental/models/` (incl. autoencoders/glue) → **36 passed**, no regressions
- Broader regression (`tests/experimental/models/` + `tests/recipes/`) → **216 passed**
- All under `JAX_PLATFORMS=cpu` with xdist (`uv run pytest …` or `mamba activate gensbi`).

---

## Part B — Before Phase 2

Ordered by priority. The first two are gates: don't trust the model or build Phase 2 on top of it until they're resolved.

### B1. Prove the model actually learns (the real "does it work")  ⚠️ GATE

**What the Phase-1 tests prove:** the model is *well-formed* — correct shape, finite, exactly zero at init, differentiable (output path connected), exports wired.

**What they do NOT prove:** that the condition influences the output, or that gradients flow through the encoder and transformer core. This is inherent to a zero-init design: at init `v == 0` for *all* inputs, and the zero-init `conv_out` multiplicatively gates every upstream gradient — empirically, **2 of 151 parameter gradients are nonzero at init; the other 149 are identically zero, by design.** No test at init can distinguish a live conditioning path from a dead one.

**Action:** add a minimal training-step check before anything else in Phase 2:
- After one optimizer step on a trivial objective, assert `v` is no longer identically zero and that gradients are nonzero for representative encoder / core / decoder-block params (not just `conv_out`).
- A tiny overfit test (drive the loss down on a handful of (field, cond) pairs) to confirm conditioning genuinely shapes the output.

Until this passes, "all tests green" means "well-formed," not "works."

### B2. Pipeline wiring  ⚠️ GATE for any training

`FieldDiT` is callable and differentiable but not connected to a `GenerativeMethod` / `ConditionalPipeline`. Per the plan's forward-looking notes (verify against current code when planning):
- `FMLoss` calls `model(obs=x_t, t=t, **model_extras)` and compares to `path_sample.dx_t` elementwise — FieldDiT's field-shaped I/O is already compatible.
- **But** `ConditionalPipeline` flattens `dim_obs` to a token count and sets `event_shape=(dim_obs, ch_obs)`, which is **wrong for pixel-space fields**. The clean follow-on is a thin field pipeline: override `event_shape = (H, W, C)`, skip id-resolution (ids are built internally), pass `cond` raw.
- `ConditionalWrapper._expand_dims` only acts when `ndim < 3`, so a 4D field `(B,H,W,C)` passes through untouched — the wrapper may be reusable as-is. **Verify this.**

### B3. Design risk — normalization vs. field statistics  🔬 STUDY

Open spec flag (§2, §9 Q1): `GroupNorm`'s spatial pooling may wash out exactly the spatial statistics an emulator must preserve (e.g. power spectrum, variance structure). Worth an **ablation** (GroupNorm vs. alternatives, or none) before scaling to 256². This is a correctness risk for the emulator face of the model, not just a tuning knob.

### B4. Design risk — does coarse conditioning shape fine scales? (R2 / flagged-C)  🔬 STUDY

The architecture bets that conditioning injected only at the coarse meeting grid can still control fine output scales, via per-decoder-stage modulation acting as per-frequency-band amplitude control (flagged-C, default ON, decoder-only). **This is a hypothesis, not a verified property.** Phase-2 validation should measure the output power spectrum as a function of the condition to confirm the fine-scale response is real.

### B5. Carry-forward technical notes (cheap, do when relevant)

- **cond batch broadcasting:** `MMDiTCore` intentionally does NOT broadcast `cond_tokens` from batch 1 → B (unlike the Flux1 reference). In Phase 1 the condition always arrives at batch B from the embedder, so it's fine. If Phase 2 introduces a batch-1 cond path or CFG, add the broadcast.
- **`ScalarCondEmbedder` 2D input** is only valid when `cond_in_channels == 1` (documented in its docstring, no runtime guard). If multi-channel cond tokens ever use the `(B, k)` shorthand, it errors cryptically — consider a guard then.
- **Guidance / CFG:** only the minimal `guidance_embed` plumbing hook exists (an MLP that adds to `vec`); there is no classifier-free-guidance sampling logic. Phase-2 sampling will need it.
- **Real-size / performance check:** all tests use tiny configs (hidden=16). The defaults are `num_heads=12, axes_dim=[16,24,24] → hidden=768`. Instantiate a realistic 256² config once to check memory/throughput and that the derived token count is what you expect before committing to a training run.
- **Differentiability test caveat:** the existing `test_fielddit_is_differentiable` only asserts the *output path* is connected (nonzero `conv_out` grad) — see B1 for why it can't do more at init.

### B6. Validation track — GRF 256²  🔬 SEPARATE PLAN (greenfield)

Per spec §6: GRF (Gaussian Random Field) 256² validation — power-spectrum recovery and field-space SBC/TARP. **There is no existing `grf.py`** — this is greenfield and belongs in its own experimental plan, distinct from both Phase 1 and the Phase-2 architecture work.

### B7. Phase 2 scope proper — image / spatially-aligned conditioning

Deferred and **purely additive** (Phase-1 architecture unchanged). Kontext-style co-tokenization: concat obs+cond tokens on a shared grid separated by a RoPE **semantic id** (`init_ids_2d(semantic_id=...)` / `init_ids_joint` already exist). Known challenges from the spec to design around: a shared tokenizer for obs and cond, read-only conditioning (cond gets no decoder/skip path), and CFG over image conditions.

---

## One-line status

Phase 1 produces outputs of the **correct nature** and is cleanly structured and tested. It is **not yet validated as a working generative model** — B1 and B2 are the gates that turn "well-formed" into "trainable and verified."
