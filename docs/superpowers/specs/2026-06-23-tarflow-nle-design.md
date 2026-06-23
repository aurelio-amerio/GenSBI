# TarFlow → GenSBI — Design (transformer autoregressive flow for NLE/NPE)

*Status: approved design, ready for implementation planning.*
*Date: 2026-06-23.*

A transformer autoregressive normalizing flow for GenSBI, added as a new density
model in the existing `normalizing_flows/` track alongside the MAF/NSF flows.
Reference algorithm: TarFlow — *Normalizing Flows are Capable Generative Models*
(Zhai et al. 2024, arXiv:2412.06329). Source notes:
`docs/superpowers/notes/tarflow_port_handout.md`. Reference implementation:
`reference/ml-tarflow` (PyTorch, Apple; "All Rights Reserved" — **clean-room
reimplementation**, no code vendored, paper cited).

## 1. Goal & scope

A conditional normalizing flow with **exact, fast, single-pass density
evaluation** that scales to higher-dimensional data than MAF+MADE, by replacing
the MADE conditioner with a **causally-masked transformer** whose autoregression
runs over a *sequence of tokens*. Primary role: the NLE likelihood estimator
`q(x | θ)`; the density evaluated inside NUTS to get the posterior.

The model is built so that working with a different kind of data (vector, time
series, image) is, as far as possible, only a change of *embedding* — the
transformer always sees `(B, T, F)` tokens. This mirrors how Flux1/FieldDiT
generalize across data via id-embeddings.

Two phases (the v1 ↔ v2 line is §10):

- **v1** — vector x, vector θ, exact `log_prob`, plus a minimal sequential
  sampler. Serves **NLE** and **vector→vector NPE**.
- **v2** — image x (patchify-2d/rope2d), arbitrary embedded θ via
  cross-attention (e.g. CNN for an image condition), KV-cache sampling, and the
  bf16/cuDNN-flash performance pass.

### Non-goals (deferred, noted but not built)
- KV-cache, guidance/CFG, `attn_temp`, annealed guidance (reference
  sample-quality / speed machinery — most never wanted for SBI).
- Noise augmentation / uniform dequantization (biases the *density*; our
  simulator outputs are already continuous — see §8).
- Deep-shallow capacity split (STARFlow); learnable per-token base variance
  (a vp-mode/sampling artifact — we use nvp with a fixed `N(0,I)` base).
- bf16/cuDNN-flash attention (v2 perf pass — large surface, deferred).

## 2. Key architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Placement | `src/gensbi/normalizing_flows/transformer_flow/` | Production NF track, per user. (Whole NF submodule may relocate to experimental later — separate, deferred decision.) |
| Relationship to MAF track | **Sibling** to `Flow`/`Chain`/`Bijection`, batched-native `(B,T,F)`; NOT built on `Chain` | TarFlow's autoregression is over the *token/sequence* axis via attention, not the per-dim MADE conditioner. Same duck-typed surface (`log_prob`, `set_standardization`, `sample`) so pipelines treat both uniformly. |
| Build strategy | **Bespoke flow-math core + reused GenSBI primitives + swappable seams** (Approach 3) | Faithful & numerically-verified exactly where the Jacobian lives; reuse `attention()`/`MLPEmbedder`/pos-embed/`apply_rope` where exactly equivalent; tokenizer/conditioner as seams for v2. Avoids the FieldDiT-style hybridization risk in the math-critical path. |
| Conditioning (v1) | **MLP(θ) broadcast-added per token** | Continuous analog of the reference `class_embed` add; cheapest conditioning that preserves the triangular Jacobian. See §6. |
| Tokenization (v1) | **Invertible reshape**, scalar-per-token default | x is the *modeled* variable, so its tokenizer must be a fixed invertible map (logdet 0), never a learned encoder. See §5. |
| Base | Fixed `N(0, I)` over `(T,F)` (nvp mode) | nvp gives the proper tractable Jacobian; standard-normal base, no learnable variance. |
| NLE ↔ NumPyro | Reuse existing `NLEPosterior` (potential fn + NUTS) | Unchanged; needs only `flow.log_prob(x, θ)`. |
| Training | Reuse existing `ConditionalFlowPipeline` (direction-agnostic max-likelihood `q(obs|cond)`) | No new pipeline. NLE: obs=x, cond=θ. |

### The asymmetry that drives the design: modeled x vs condition θ

In `q(x|θ)` the two inputs play different mathematical roles, and "just change
the embedding" is safe for one and dangerous for the other:

- **θ (condition)** is not assigned a density — it only feeds the affine params.
  Its embedder may be **anything** (MLP, CNN, patchify+rope, multi-token
  memory). None of it touches the Jacobian.
- **x (modeled)** must keep a **triangular Jacobian**. Its embedding must be a
  *fixed, invertible tokenizer + positional ids + a causal token order*, NOT a
  learned lossy encoder. Positional ids are added to the hidden stream
  (`proj_in(x)+pos`), never to x, so they never enter the change-of-variables.

This is why image **conditions** (v2, CNN) are easy, while image **x** (v2)
stays an invertible patchify, and why symmetric MM-DiT joint attention is
rejected (§6).

## 3. Module layout

```
src/gensbi/normalizing_flows/
  transformer_flow/
    __init__.py        # exports TransformerFlow, make_tarflow
    model.py           # TransformerFlow + make_tarflow
    blocks.py          # MetaBlock, AttentionBlock (causal)
    tokenizers.py      # Tokenizer interface + VectorTokenizer
    conditioners.py    # Conditioner interface + VectorConditioner
  flow.py, bijections/ # existing MAF/NSF track (untouched)
```

`TransformerFlow` and `make_tarflow` are re-exported from
`gensbi.normalizing_flows`. **Naming:** class `TransformerFlow` (descriptive,
license-clean), builder `make_tarflow`.

## 4. Components

- **`AttentionBlock`** — pre-norm residual transformer block:
  `LayerNorm → attention(is_causal=True) → +x`, then `LayerNorm → MLP → +x`.
  Reuses GenSBI's mask-aware `attention()` (`models/flux1/math.py`) and an MLP.
  fp32 / XLA in v1 (`is_causal=True` is correct now and flash-ready later).

- **`MetaBlock`** (the faithful flow core — one exact bijection over tokens):
  1. permute token order (+ pos-ids);
  2. `proj_in: F→channels`; add pos-embed; add conditioning signal (§6);
  3. `num_layers` causal `AttentionBlock`s;
  4. `proj_out: channels→2F`, **zero-initialised** (identity warm-start),
     giving per-token-feature `(a, b)`;
  5. **shift-by-one** (`[zeros, out[:-1]]`) so token *i*'s params depend only on
     tokens `< i`;
  6. forward (data→noise): `z = (x − b)·exp(−a)`, **logdet `= −Σ a`** over tokens
     & features (accumulated in fp32); un-permute.
  - `reverse` (noise→data): sequential scan over tokens, re-running the causal
    forward on the partially-built sequence to recover each `x_i = z_i·exp(a_i)
    + b_i`; un-permute. Structurally identical to the existing
    `MaskedAutoregressive.forward` scan. No KV-cache in v1 (O(T²)/step — fine for
    low-D).

- **`TransformerFlow`** (model; analog of the reference `Model` and of `Flow`):
  holds the tokenizer, conditioner, a list of `MetaBlock`s with alternating
  permutations, the `N(0,I)` base over `(T,F)`, and an optional raw-x
  standardization (logdet `−Σ log std`). Batched-native, jit/grad-friendly.
  - `log_prob(x, cond=None) → (B,)`: standardize → tokenize → embed cond → run
    `MetaBlock`s (data→noise) → `base.log_prob(z) + Σ logdet`.
  - `sample(key, cond=None, nsamples=None) → x`: signature matching
    `Flow.sample` so `ConditionalFlowPipeline.sample`/`sample_batched` work
    unchanged. base sample → reverse the `MetaBlock`s → detokenize →
    un-standardize.
  - `set_standardization(mean, std)`.

- **`make_tarflow(rngs, dim, cond_dim=0, channels=256, num_blocks=8,
  layers_per_block=2, head_dim=64, block_size=1, permutation="flip",
  standardize=True, zero_init=True)`** — builder mirroring `make_maf`.

## 5. Tokenizer (seam)

Interface: `tokenize(x) → (tokens (B,T,F), ids)` and `detokenize(tokens) → x`,
with a documented/asserted **logdet 0** (volume-preserving).

- **v1 `VectorTokenizer(dim, block_size=1)`** — reshape `(B, dim) → (B, T, F)`,
  `T = dim/block_size`, `F = block_size`. Causal order = dim order; 1-d
  positional ids (`arange T`). `block_size>1` (block-per-token) is a config knob
  to cap sequence length for larger x (attention is O(T²)); default scalar-per-
  token (finest autoregression).
- **v2 `ImageTokenizer`** — patchify-2d (invertible reshape) + rope2d ids,
  raster causal order. Same interface; logdet still 0.

## 6. Conditioner (seam) & conditioning mechanism

Interface: `embed(cond) → signal`, `inject(tokens, signal) → tokens`,
`cond=None` → unconditional.

- **v1 `VectorConditioner(cond_dim, channels)`** — "MLP per-token add": a
  per-`MetaBlock` `MLPEmbedder(cond_dim → channels)` produces a `(B, channels)`
  vector, broadcast-added to every token after `proj_in+pos`:
  `h = h + cond_mlp(cond)[:, None, :]`. The affine params `(a,b)` thereby depend
  on θ, so `q(x|θ)` depends on θ. Per-block MLP (mirrors the reference's
  per-block `class_embed`); a shared embedder is the cheaper alternative.
  - **Exactness:** the signal depends only on θ (constant w.r.t. x) and is added
    identically to all tokens, so it shifts the params without creating any
    `x_{>i} → params_i` dependence — triangular Jacobian preserved.

- **v2 `CrossAttnConditioner`** — θ encoded by an arbitrary embedder
  (MLP/CNN/patchify+rope) into **memory tokens**; x cross-attends to them.
  Required for *spatially structured* conditions (an image) that a single bias
  vector cannot represent.
  - **Why not symmetric MM-DiT:** if θ-tokens attended to x-tokens, a θ-token
    would become a function of *all* x; an earlier x-token attending to it would
    then indirectly see later x-tokens → Jacobian no longer triangular → the
    `−Σa` log-det is silently wrong. The safe pattern (x→x causal, x→θ full,
    θ→x blocked) collapses to cross-attention to a θ-memory, which is also
    cheaper. Symmetric joint attention is rejected.

## 7. Data flow

**NLE training** (existing `ConditionalFlowPipeline`):
```
dataset → (obs=x, cond=θ)  each (B, dim, 1)
  → _squeeze_ch → (B, dim_x), (B, dim_θ)
  → loss = −mean(model.log_prob(x, θ))
  → AbstractPipeline.train: EMA + checkpointing
```
Call `pipeline.fit_standardization(x_train)` **before** `train()` (sets the
standardize buffer on both `model` and `ema_model`; for NLE the modeled variable
is x, so stats are fit on x).

**NLE inference** (existing `NLEPosterior`):
```
NLEPosterior(flow=trained_tarflow, prior)
  U(θ) = −(flow.log_prob(x_o, θ) + prior.log_prob(θ))
  → NUTS samples θ
```

**Vector→vector NPE** (bonus from `sample()`): train with obs=θ, cond=x;
`ConditionalFlowPipeline.sample`/`sample_batched` call `TransformerFlow.sample`
unchanged.

## 8. Exactness invariants (must not regress)

1. **Triangular Jacobian** — token *i*'s params depend only on tokens `< i`
   (causal mask + shift-by-one) ⇒ `log|det| = −Σa`. Verified against an autodiff
   Jacobian.
2. **Norm over the channel axis only — never the sequence axis.** The
   autoregressive rank here is the *token* axis; LayerNorm over a token's
   *channels* is safe, but any normalization mixing *across tokens* leaks future
   tokens into earlier ones and silently breaks the density. (TarFlow analog of
   the MADE cross-feature-norm gotcha.)
3. **Tokenizer volume-preserving** (reshape ⇒ logdet 0); x's tokenizer never a
   learned lossy encoder.
4. **Identity warm-start** — `zero_init` proj_out ⇒ each `MetaBlock` is identity
   ⇒ `log q(x|θ)` starts as the standardized-Gaussian base.
5. **No noise augmentation / dequantization** — would learn the density of
   noised x ⇒ biased, over-smoothed likelihood ⇒ biased posteriors. Default off;
   document if ever used.

## 9. Testing & validation

Run with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest` (GPUs usually busy).

**Fast unit tests**
- **logdet vs autodiff Jacobian** (small dim): analytic `−Σa` matches `jacrev`
  `log|det|` of the data→noise map to ~1e-5. *(exactness)*
- **Triangular/causal structure:** `∂z_i/∂x_j == 0` whenever *j* is after *i* in
  each block's causal order (with the permutation). *(no-leakage guard)*
- **Round-trip bijectivity:** `detokenize(reverse(forward(x))) == x`. *(catches
  shift-by-one / permutation / standardization-ordering bugs)*
- **Identity warm-start:** `zero_init` ⇒ `z == standardized x`, flow logdet
  `== 0`, `log_prob == base.log_prob(standardized x)`.
- **Tokenizer round-trip** and logdet 0.
- **Conditioning is real and normalized:** `q(x|θ₁) ≠ q(x|θ₂)`, and a 1-D
  numerical integral of `exp(log_prob)` ≈ 1.
- **Unconditional path** (`cond=None`) works.

**Slow tests** (`@pytest.mark.slow`)
- **Linear-Gaussian recovery:** train an NLE `TransformerFlow` on a
  linear-Gaussian simulator; learned `log q(x|θ)` tracks the analytic
  likelihood, and `NLEPosterior` recovers the analytic posterior. Mirrors the
  existing MAF recovery test.

## 10. The v1 ↔ v2 line

| | **v1 (this spec)** | **v2 (separate plan)** |
|---|---|---|
| modeled x | vector (reshape tokenizer, 1d ids) | + image (patchify-2d, rope2d) |
| condition θ | vector (MLP, per-token add) | + arbitrary object (CNN/patchify) via cross-attention |
| direction | NLE **and** vector→vector NPE | NPE with embedded conditions |
| sampling | minimal sequential reverse + `sample()` (no cache) | KV-cache; sampling under embedded-condition cross-attn |
| attention | XLA / fp32, `is_causal=True` | + bf16/cuDNN-flash perf pass |

Both seams (tokenizer, conditioner) and the `is_causal`/`implementation` knobs
ship in v1, so v2 is extension, not rearchitecture.

## 11. Integration points (touch vs. reuse)

- **Reuse unchanged:** `ConditionalFlowPipeline` (train loop, EMA, checkpointing,
  standardization), `NLEPosterior` (NUTS), `diagnostics/` (SBC/TARP/LC2ST), and
  low-level primitives `attention()`, `MLPEmbedder`, pos-embed, `apply_rope`.
- **New code only** in `normalizing_flows/transformer_flow/`.
- **Known boundary limit:** `_squeeze_ch` assumes `ch==1` / `(B, dim)` — fine for
  vector x; structured x in v2 needs a pipeline-boundary adapter (flagged, not
  built now).
