# Mixed-Precision Training — Implementation Handout

**Branch:** merged into local `main` (`a8b4ee7..8460091`, 18 commits, fast-forward). Not pushed.
**Verification:** full suite on merged main: **1131 passed, 2 pre-existing opt-in skips, 0 failed** (CPU).
**Plan:** `docs/superpowers/plans/2026-07-20-mixed-precision.md` · **Spec:** `docs/superpowers/specs/2026-07-20-mixed-precision-design.md`

---

## The contract (what every model now guarantees)

| Rule | Implementation |
|---|---|
| **fp32 master weights** | `param_dtype=jnp.float32` everywhere; zero `param_dtype=bfloat16` left in `src/` (swept) |
| **Compute knob** | every `*Params` dataclass has `dtype` (compute/matmul dtype), threaded into every `nnx.Linear`/conv |
| **fp32 islands** (always, knob-independent) | norm layers, attention softmax, timestep/sinusoidal/Fourier embeddings, RoPE, final projection, loss |
| **Models emit fp32** | final projection **constructed** with `dtype=jnp.float32` — never a post-hoc `.astype` |
| **Inputs fp32 at the door** | all `jnp.asarray(x, param_dtype)` input casts replaced with `jnp.asarray(x, jnp.float32)` |
| **Losses fp32** | model output + target upcast to fp32 before weighting/reduction (defense-in-depth) |
| **Grads/AdamW/EMA fp32** | automatic — JAX gives gradients the params' dtype (test-verified per model) |
| **No loss scaling** | bf16 shares fp32's exponent range |

### Per-model defaults

| bf16 compute by default | fp32 by default (knob present, flip later) |
|---|---|
| Flux1, Flux1Joint, Simformer, PixelDiT, FieldDiT, autoencoders (1D/2D) | MAF, TarFlow — **bit-identical** to pre-refactor code at the default (golden `jnp.array_equal` tests) |

`gensbi.models.healswin` untouched (external package, out of scope).

---

## What changed, by commit

| Commit | Area | Notes |
|---|---|---|
| `2a16345` | `tests/precision_utils.py` | shared `assert_tree_dtype` / `float_leaves` helpers |
| `75e6734` | losses | fp32 casts in `FMLoss` (`fm_loss.py`), EDM (`path/scheduler/edm.py`), SM (`path/sm_path.py` — the arithmetic lives in the path closures, not the `loss/` wrapper files the plan named), NF pipeline `-log_prob` scalar |
| `7713355`, `cef55a2` | shared embeddings | `MLPEmbedder`/`FeatureEmbedder`/`GaussianFourierEmbedding` accept `dtype` (fp32-neutral default); **fix:** GFE trig now always fp32, only the output is cast |
| `f33d9e0` | Flux1 | params + all 7 layer classes in `flux1/layers.py`; QKNorm re-casts normed q/k to `v.dtype`; `LastLayer.linear` fp32 |
| `d14c8ba`, `2db7ef6` | Flux1Joint | **fix:** `condition_embedding` cast to compute dtype at use site — uncast fp32 Param was re-promoting the whole single-block stack |
| `3bd0771`, `4258896` | Simformer | fp32 `AttentionBlock` island (flax `nnx.MultiHeadAttention` softmax runs in its `dtype`, so the whole block is the island); **fix:** blocks downcast their *output* to compute dtype — island = fp32 math, emit compute dtype |
| `5b11ad1` | MAF | knob threaded through MADE/MaskedLinear; log-det arithmetic untouched; golden bit-identical |
| `266d2e3` | TarFlow | knob through blocks/conditioners; softplus/soft_clip scale path, log-det sums, mean/std buffers, KV-cache dtype behavior all untouched; golden bit-identical + 18-test cached≡reference gate green |
| `d88fb92` | PixelDiT | canonical pattern; caught `CondTokenEmbedder` norm-output leak; 2 stale tests pinned fp32 (they assert tighter-than-bf16 numerics) |
| `1c34f7f`, `830bf7e` | FieldDiT | reuses Task-4 flux1 layers; caught `cond_ids_embedder` leak; closeness test un-gates the MMDiT core (AdaLN-zero made it an identity otherwise) |
| `26ac69f` | Autoencoders | 1D/2D threaded; `DiagonalGaussian` sampling fp32; `encode`/`decode` scale/shift Params cast at use site |
| `dc9ebb9`, `0a2ac77` | Pipeline | `_warn_if_not_fp32_master_weights` guard in `Pipeline.__init__`; EMA regression tests; orbax `restore_model` casts restored leaves to target dtypes (model + EMA paths); safetensors bf16→fp32 load regression lock; **the guard caught a live straggler:** `recipes/flux1joint.py` still defaulted bf16 |
| `8460091` | final-review fixes | see below |

---

## The recurring bug class this branch discovered (worth knowing)

**An fp32 `nnx.Param` or fp32-island output entering a bf16 stream silently re-promotes every downstream residual to fp32**, defeating the knob with zero test signal at the endpoints (output is fp32 either way).

- Self-heals only when the fp32 value feeds a compute `nnx.Linear`/conv (its `promote_dtype` downcasts). Residual adds and concats do **not** self-heal.
- Found and fixed in **five** models: Flux1Joint (`condition_embedding`), Simformer (block outputs), PixelDiT (`CondTokenEmbedder`), FieldDiT (`cond_ids_embedder`), Flux1 (`pos1d`/`pos2d` id-embeddings — final review).
- Each fix is guarded by a **spy test**: class-level monkeypatch capturing the dtype of the activation entering an inner block under `dtype=bf16`, RED-verified against the pre-fix code.

**Idiom to remember:** storage stays `param_dtype`; cast at the use site (`.astype(stream.dtype)`). Islands do fp32 *math* but emit the compute dtype — except the designated emit-fp32 endpoints (final projections).

## The EMA bug, now test-documented

`tests/recipes/test_precision_pipeline.py`:
- fp32 EMA (decay 0.999, 0.1%-scale updates, 500 steps) tracks a float64 reference to ~9e-6.
- The same accumulation in bf16 diverges by ~0.0215 — **~2500× worse** — because the `(1-decay)·w` increment is below bf16's mantissa resolution. This is the white-noise-EMA mechanism.
- AdamW moments verified fp32 once params are fp32.
- Gotcha found on the way: `optax.ema`'s raw accumulator **zero**-initializes; the plan's reference recursion started at 1 and had to be corrected (the bf16 counter-test was non-diagnostic until then).

---

## Final whole-branch review → `8460091`

Verdict was "with fixes"; all three fixed and re-verified, final verdict **ready to merge**:

1. **Flux1 id-embed leak** — `pos1d`/`pos2d` FeatureEmbedder outputs (fp32 islands) now cast to stream dtype at all four merge points + spy test.
2. **YAML knob** — `dtype:` now parsed in `recipes/flux1.py`, `recipes/flux1joint.py`, `experimental/recipes/vae_pipeline.py` (default `"bfloat16"`). Config users can opt out of bf16 compute.
3. **TarFlow norms** — `norm1`/`norm2` pinned to fp32 islands (was knob-following, contradicting the Global Constraints). Bit-identical golden unchanged — provably a no-op at the fp32 default.

---

## How to check it yourself

```bash
# full suite (mamba gensbi env; ~20 min CPU)
JAX_PLATFORMS=cpu python -m pytest tests/ -q

# just the precision tests
JAX_PLATFORMS=cpu python -m pytest tests/test_precision_utils.py tests/models/losses/ \
  tests/models/embedding/test_embedding_dtype.py -q \
  -k "precision or dtype" tests/models tests/experimental tests/recipes tests/utils

# sweep: must return nothing
grep -rn "param_dtype: DTypeLike = jnp.bfloat16\|param_dtype=jnp.bfloat16" src/gensbi/
```

Per-task implementation/review reports were session-scratch in the (now removed) worktree's `.superpowers/sdd/` — the durable record is the commit history above.

### GPU validation gates (yours, the real test)

1. **PixelDiT GRF probe with `use_ema=True`** — expect structure, not white noise. This is the direct test of the EMA-bug hypothesis.
2. **Flux1 two-moons sanity run** — mixed-precision convergence should be ≥ the old fp32-parity baseline.
3. Old checkpoints: bf16-master-weight checkpoints load into the new fp32 models (both safetensors and orbax cast on restore) — a quick restore of a real old checkpoint would be a nice extra check.

---

## Known gaps / deliberate leftovers (all judged non-blocking in review)

- Plan Step 13.3 (recovery-script smoke) was **impossible**: `scripts/maf_nle_recovery.py` + tarflow twin don't exist on any branch — only `docs/notebooks/two_moons_maf_nle.ipynb` / `slcp_tarflow_nle.ipynb`. The plan doc (and old session notes) reference stale paths; reconcile when convenient.
- MAF/TarFlow bf16 flip is a **future** step (knob exists, fp32 default) — per the spec, flip after testing.
- MAF's `MaskedLinear` passes `dtype` to an internal `nnx.Linear` whose `__call__` is bypassed (hand-rolled matmul does the real cast) — harmless, could drop the kwarg.
- Flux1Joint `id_merge_mode="concat"` has no dedicated dtype test (the fix is shared code ahead of the branch split).
- Autoencoder bf16-closeness tolerance is 5e-2 (measured 2.6% over a ~9-block stack); could tighten to ~3.5e-2.
- No docs page yet for the `dtype`/`param_dtype` split — the most user-visible behavior change (bf16 compute is now the *default* for six models); a short note in `docs/advanced/` would be worthwhile.
