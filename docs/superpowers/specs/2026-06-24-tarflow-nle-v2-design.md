# TarFlow → GenSBI v2 — Design (structured x & structured conditioning)

*Status: approved design, ready for implementation planning.*
*Date: 2026-06-24.*

Extends the v1 `TransformerFlow` (`src/gensbi/normalizing_flows/transformer_flow/`)
to fill in the two seams v1 deferred, so the same model serves:

- **Field-level NLE** — the modeled variable `x` is an image/field
  (`q(x_img | θ)`), via a new invertible `ImageTokenizer`.
- **Structured / rich conditioning** — the condition (a vector θ *or* an image
  observation `x`) enters via **prefix-concatenation** (ported from STARFlow),
  not the v1 per-token additive bias and not cross-attention.

v2 is a pure extension: it does **not** change the Jacobian-critical flow math,
the v1 exactness invariants, or the existing vector path. References:
`reference/ml-tarflow/transformer_flow.py` (the v1 base) and
`reference/ml-starflow/transformer_flow.py` (the conditioning mechanism).
v1 spec: `docs/superpowers/specs/2026-06-23-tarflow-nle-design.md`.

## 1. Goal & scope

| | **v1 (shipped)** | **v2 (this spec)** | **v3 (deferred)** |
|---|---|---|---|
| modeled x | vector (reshape tokenizer) | **+ image (`ImageTokenizer`, patchify)** | — |
| condition | vector, additive bias (`VectorConditioner`) | **+ prefix-concat: vector or image condition as prefix tokens** | — |
| first token | unconditioned (zero-shift) | **conditioned (SOS input-shift)** | — |
| position | learned `pos_embed` | learned `pos_embed` (+ a learned prefix pos-embed) | rope2d/3d |
| sampling | sequential reverse, no cache | sequential reverse, no cache | KV-cache |
| attention | XLA / fp32, `is_causal=True` | XLA / fp32, **prefix-causal mask** | bf16/cuDNN-flash |

### Non-goals (deferred to v3, noted but not built)
- **KV-cache sampling.** In both v2 use cases the sequential sampler is off the
  critical path: field-NLE does inference through `log_prob` (the parallel
  data→noise pass; NUTS samples θ, not x), and image-NPE has a tiny modeled θ
  (small T). KV-cache only speeds *sampling*. STARFlow's cache is built for the
  prefix layout, so v3 inherits it cleanly.
- **bf16 / cuDNN-flash attention.** Throughput, not correctness; touches every
  block. Large surface, deferred.
- **rope2d/3d positions.** A modeling-quality knob (positions never touch the
  change-of-variables — see §8), and STARFlow's 3D rope reserves an axis for a
  *text* prefix. **v3 prerequisite:** define a positional scheme for an
  arbitrary (non-text) SBI condition prefix before adopting rope. Infra exists
  (`models/flux1`: `EmbedND`, `apply_rope`).
- **Symmetric MM-DiT joint attention** — rejected: a condition token attending
  to x would make the prefix a function of x and break triangularity (§2, §8).
- **CFG / guidance / `attn_temp` / annealed guidance** — sample-quality
  machinery; not wanted for SBI.

## 2. Why prefix-concat, not cross-attention (the lineage argument)

The v1 spec §6 assumed structured conditioning would arrive as **cross-attention
to a condition "memory."** Reading the references changed this decision:

- **TarFlow (v1 base) has no cross-attention.** Its only conditioning is an
  additive class-embedding for *discrete ImageNet labels*
  (`ml-tarflow:139–167`): a scalar id has no structure to attend to, so an
  additive vector is the minimal sufficient mechanism. The v1 `VectorConditioner`
  is the continuous analog — faithful, and kept in v2.
- **STARFlow (the newer model) has no cross-attention either.** It handles rich
  *text* conditioning by **prepending the condition tokens as a causal prefix**
  to the modeled tokens and running one causal self-attention over `[cond ; x]`
  (`ml-starflow:331` `cat([y, x], dim=1)`, `:439` `tril(M+T)` mask, `:339` split
  back off). The condition is projected to model width (`proj_txt`, `:418`) and
  becomes additional *sequence positions* — not extra features on each token.

The prefix-concat pattern is **exactly the "safe pattern" v1 §6 named**
(`x→x causal, x→cond full, cond→x blocked`), but realized via masking instead of
a separate attention module. v1 claimed that pattern "collapses to
cross-attention to a memory"; it *also* collapses to prefix-concat, which is what
the SOTA successor actually uses. We adopt prefix-concat because:

1. **Faithful to the lineage** — it ports STARFlow's real mechanism; cross-attn
   is a from-scratch GenSBI invention neither reference uses.
2. **Smaller math-critical surface** — reuses the existing self-attention; the
   only change is the attention *mask*, not a new attention type with its own
   QKV/norm.
3. **Conditions the first token** via an SOS input-shift (§7), fixing v1's
   unconditioned token 0 — which matters for the small modeled vectors common in
   SBI.
4. **Same Jacobian guarantee** — the prefix is condition-only and causally
   upstream, so token *i*'s affine params still depend only on modeled tokens
   `< i` (§8).

Cross-attention's one genuine advantage — decoupling the condition size from the
modeled sequence length (the memory is computed once, independent of T) — only
pays off when the condition is huge and the modeled variable tiny. We accept the
prefix's `O((M+T)²)` attention cost as the price of the smaller, faithful
surface; cross-attention remains a possible future alternative if an `M ≫ T`
workload demands it.

## 3. Key architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Conditioning mechanism | **Prefix-concat** (port STARFlow) | §2. Reuse self-attention; faithful; conditions token 0. |
| First-token handling | **SOS input-shift** for the conditioned path | Token 0 sees the prefix; helps small-θ SBI. v1 zero-shift kept for the unconditional path. |
| Image modeled var | **`ImageTokenizer`** over `recipes.utils.patchify_2d` | Invertible reshape, logdet 0; raster causal order; existing learned `pos_embed` handles position. |
| Condition encoders | **patch-embed (image) + linear/MLP (vector)** → prefix tokens | One encoder family; upstream CNN/VAE is a user choice (lensing-example pattern). |
| Position | learned `pos_embed` (modeled) + learned prefix pos-embed | rope2d/3d deferred (§1); positions never touch the Jacobian (§8). |
| Pipeline boundary | **generalize `ConditionalFlowPipeline` in place** with `structured_obs`/`structured_cond` flags (default off) | Vector/MAF path byte-identical; structured side bypasses `_squeeze_ch`. |
| NLE with structured x | generalize `NLEPosterior` to pass a structured `x_o` through | Stops the `atleast_1d(squeeze(...))` flatten; NUTS still samples the θ vector. |
| Base, training, NUTS | unchanged from v1 | nvp `N(0,I)`; `ConditionalFlowPipeline`; `NLEPosterior`. |

## 4. The two seams compose orthogonally

The **tokenizer** (what the modeled variable is) and the **conditioner** (how the
condition enters) are independent axes. All four combinations are valid; the
Jacobian argument (§8) holds for every cell because the conditioning signal is
always a function of the condition only.

| | additive bias (`VectorConditioner`, v1) | prefix-concat (`PrefixConditioner`, v2) |
|---|---|---|
| **vector modeled var** (`VectorTokenizer`) | v1 — vector NLE/NPE | vector or image condition via prefix |
| **image modeled var** (`ImageTokenizer`, v2) | field NLE, cheap vector-θ bias | field NLE / image-condition, prefix |

Typical v2 uses:
- **Field NLE** `q(x_img | θ)`: `ImageTokenizer` + (`VectorConditioner` *or*
  `PrefixConditioner` with a vector θ).
- **Image NPE** `q(θ | x_img)`: `VectorTokenizer` (θ small) + `PrefixConditioner`
  with the image `x` patch-embedded into prefix tokens.

## 5. `ImageTokenizer` (modeled-variable seam) — `tokenizers.py`

Thin wrapper over `gensbi.recipes.utils.patchify_2d` / `depatchify_2d`:

- `tokenize(x: (B, H, W, C)) -> (B, T, F)`, `T = (H/p)·(W/p)`, `F = C·p²`,
  raster causal order (the `rearrange` in `patchify_2d` fixes the order).
- `detokenize(tokens) -> (B, H, W, C)` via `depatchify_2d(grid=(H/p, W/p))`.
- Same interface as `VectorTokenizer`; **asserted logdet 0** (pure reshape).
- Position handled by the existing `MetaBlock` learned `pos_embed` over the T
  patches — no rope (deferred, §1).

Constructor: `ImageTokenizer(height, width, channels, patch_size)`, exposing
`.T`, `.F` like `VectorTokenizer` so `make_tarflow` and `MetaBlock` consume it
unchanged.

## 6. Conditioner: prefix-concat + encoders — `conditioners.py`

The v1 conditioner interface (`embed`/`inject`) generalizes to a small uniform
contract so a block can consume either an additive bias **or** prefix tokens:

```
embed(cond) -> ConditioningContext(bias: (B,C) | None, prefix: (B,M,C) | None)
```

- **`VectorConditioner` (v1, kept):** returns `bias` (per-token add). Unchanged.
- **`PrefixConditioner` (v2, new):** returns `prefix` — M condition tokens at
  model width `C`, each carrying a learned prefix pos-embed `(M, C)`. Two
  built-in encoders select on the condition shape:
  - **vector condition** → `Linear/MLP(cond_dim → C)` → `(B, 1, C)` (one prefix
    token; a small `M>1` knob is allowed).
  - **image condition** → `patchify_2d` → `Linear(C·p² → C)` → `(B, M, C)` with
    `M = (H/p)(W/p)` prefix tokens.
  - Any upstream CNN/VAE runs *before* the conditioner and hands its feature map
    in as `cond` (the lensing-example pattern); no encoder is baked into the NF
    track beyond patch-embed + linear.

`PrefixConditioner` exposes `.M` (number of prefix tokens) so the block can size
its attention mask.

## 7. Block & MetaBlock changes — `blocks.py`

**`AttentionBlock`:** the self-attention call moves from a hard `is_causal=True`
to an explicit **mask argument**:
- no prefix → `tril(T)` (≡ v1 `is_causal=True`).
- prefix present → `tril(M+T)`, so each modeled token attends to *all* M prefix
  tokens + earlier modeled tokens; prefix tokens attend only causally among
  themselves and **never** to modeled tokens (`ml-starflow:439`).

The mask is built once per forward from `(M, T)` and passed to
`jax.nn.dot_product_attention(q, k, v, mask=...)`.

**`MetaBlock`** (the change is mechanical; the flow math is untouched):
1. tokenize + permute modeled tokens; `proj_in` + modeled `pos_embed`.
2. **SOS input-shift** (conditioned path): prepend a learned `sos_embed` and drop
   the last modeled token (`ml-starflow:466`), so position *i*'s hidden state is
   built from modeled tokens `< i` (+ the prefix). Replaces v1's post-`proj_out`
   zero-shift for the conditioned path; the unconditional path keeps the v1
   zero-shift. **Either way the dependency structure is identical** — params for
   token *i* depend only on modeled tokens `< i`.
3. if `prefix` present: concat `[prefix ; modeled]`, run the `AttentionBlock`s
   with the `tril(M+T)` mask, then **split the prefix off** before `proj_out` —
   so only modeled tokens produce affine params `(a, b)`.
   if `bias` present (`VectorConditioner` path): add it per token (v1 behavior).
4. `proj_out` (zero-init) on modeled tokens → `(a, b)`; forward
   `z = (x − b)·exp(−a)`, **logdet `= −Σ a`** over modeled tokens & features;
   un-permute. Unchanged from v1.
5. `reverse` (sampling): sequential scan over modeled tokens, re-running the
   masked causal forward on `[prefix ; partial-x]`; the prefix is constant across
   steps. No KV-cache (v3).

## 8. Exactness invariants (extended from v1; must not regress)

1. **Triangular Jacobian.** Token *i*'s affine params depend only on modeled
   tokens `< i`. The prefix is causally upstream and condition-only, so
   `∂(params_i)/∂x_j = 0` for `j ≥ i`, while `∂(params_i)/∂cond` is unrestricted
   (that *is* the conditioning). ⇒ `log|det| = −Σ a` over modeled tokens.
   Verified against an autodiff Jacobian, with the condition held fixed.
2. **Prefix is condition-only.** Condition tokens never attend to modeled tokens
   (mask blocks `cond→x`). If they did, the prefix would become a function of all
   x and an earlier x-token would indirectly see later x → triangularity broken,
   `−Σa` silently wrong. (Why symmetric MM-DiT joint attention is rejected.)
3. **Norm over the channel axis only**, never across tokens — including across
   the prefix↔modeled boundary. (v1 invariant, unchanged.)
4. **Tokenizer volume-preserving** — `ImageTokenizer` is a `patchify_2d` reshape,
   logdet 0; asserted. Never a learned lossy encoder for the *modeled* variable.
   (Encoders for the *condition* are unconstrained — they don't touch the
   Jacobian.)
5. **Identity warm-start** — `zero_init` `proj_out` ⇒ each `MetaBlock` is identity
   ⇒ `log q(x|cond)` starts at the standardized-Gaussian base. (SOS/prefix don't
   change this: with zero affine params the map is identity regardless of the
   hidden state.)
6. **No noise augmentation / dequantization.** (v1 invariant, unchanged.)

## 9. Pipeline & inference boundary

- **`ConditionalFlowPipeline`** gains `structured_obs=False`,
  `structured_cond=False` (default off ⇒ vector/MAF path byte-identical). When a
  side is structured: it bypasses `_squeeze_ch` (the model's tokenizer/conditioner
  owns the reshape), `fit_standardization` computes per-channel (broadcast) stats
  for a structured modeled var, and `_single_cond` passes a structured `x_o`
  through instead of flattening it.
- **`NLEPosterior`** stops flattening `x_o` (`atleast_1d(squeeze(...))`) and
  passes a structured observation to `flow.log_prob(x_o_img, θ)`. NUTS still
  samples the θ vector — unchanged.
- **NPE** needs no NUTS wrapper: train `q(θ | x_img)` and call `sample()`.
- **`TransformerFlow` / `make_tarflow`** accept image modeled-var params
  (`img_size`/`patch_size`/`channels`) and/or a structured-condition spec;
  standardization buffers become broadcastable to the modeled var's shape.

## 10. Module layout (changes only)

```
src/gensbi/normalizing_flows/transformer_flow/
  tokenizers.py     # + ImageTokenizer
  conditioners.py   # + PrefixConditioner (+ ConditioningContext); VectorConditioner kept
  blocks.py         # AttentionBlock: mask arg; MetaBlock: SOS shift + prefix concat/split
  model.py          # TransformerFlow/make_tarflow: image + structured-cond wiring
src/gensbi/recipes/flow_pipeline.py   # structured_obs/structured_cond flags
src/gensbi/inference/nle.py           # structured x_o
```

## 11. Testing & validation (matches the v1 pattern)

Run with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest`.

**Fast CI invariant tests**
- `ImageTokenizer`: round-trip `detokenize(tokenize(x)) == x`; logdet 0; tokens
  match `patchify_2d` directly.
- Prefix-causal mask: with a prefix, modeled token *i*'s output is invariant to
  modeled tokens `> i` (causal), depends on the whole prefix, and prefix-token
  outputs are invariant to modeled tokens (`cond→x` blocked).
- `MetaBlock` with a prefix: logdet `−Σa` matches the autodiff Jacobian of the
  data→noise map (condition fixed); `∂z_i/∂x_j == 0` for `j > i`.
- `TransformerFlow`: `zero_init` ⇒ standardized-Gaussian base (with prefix +
  SOS); 1-D/2-D numerical integral of `exp(log_prob)` ≈ 1; `q(·|c₁) ≠ q(·|c₂)`
  for a vector prefix, an image prefix, and an image modeled var.
- Round-trip `sample`/`log_prob` finite for the image-modeled and
  image-condition paths.

**Smoke CI integration** (`ConditionalFlowPipeline`, `NLEPosterior`)
- `structured_obs` (field NLE) and `structured_cond` (image NPE): loss finite,
  grads flow to params, train 2 steps, `log_prob`/`sample` shapes, the NLE
  potential value+grad on a structured `x_o`.

**GPU recovery scripts** (smoke-only in CI; full runs by the user on GPU),
mirroring `scripts/{tarflow,maf}_nle_recovery.py`:
- `scripts/tarflow_field_nle_recovery.py` — a tiny structured linear-Gaussian
  (small field x), NLE+NUTS recovers the analytic θ posterior.
- `scripts/tarflow_image_npe_recovery.py` — an image condition that is a linear
  map of θ + noise; train `q(θ|x_img)`, recover the analytic posterior by direct
  sampling.

## 12. Integration points (touch vs. reuse)

- **Reuse unchanged:** `patchify_2d`/`depatchify_2d`, `AbstractPipeline` (train
  loop, EMA, checkpointing), NUTS in `NLEPosterior`, `diagnostics/`, and the v1
  flow core (`MetaBlock` affine, `−Σa` logdet, base, standardization).
- **Touch:** `tokenizers.py`, `conditioners.py`, `blocks.py`, `model.py` (new
  code + the mask/SOS/prefix changes), `flow_pipeline.py` (two flags),
  `nle.py` (structured `x_o`).
- **Out of scope (v3):** KV-cache, bf16/flash, rope2d/3d, cross-attention,
  CFG/guidance.
