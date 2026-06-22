# Normalizing Flows in GenSBI — Design

*Status: approved design, ready for implementation planning.*
*Date: 2026-06-21.*

Discrete (autoregressive) normalizing flows for GenSBI, alongside the existing
flow-matching and diffusion methods. Reference algorithm: MAF (Papamakarios
et al. 2017) + MADE (Germain et al. 2015), with an RQ-NSF upgrade
(Durkan et al. 2019). Source notes: `docs/superpowers/notes/maf_nle_handout.md`.
Algorithm reference implementation: `reference/flowjax` (Equinox).

## 1. Goal & scope

Add discrete normalizing flows as a **first-class generative method** supporting
both simulation-based-inference directions:

- **NPE** — amortized posterior estimation `q(θ | x)`; sample the posterior
  directly. Enables a clean apples-to-apples comparison against the existing
  flow-matching NPE (`ConditionalPipeline`).
- **NLE** — amortized likelihood estimation `q(x | θ)`; evaluate inside NUTS to
  get the posterior.

Both **amortized** (single round). This document is the full A+B+C design; the
build is **phased** (§10).

### Non-goals (deferred, noted but not built)
- Sequential rounds (SNPE/SNLE).
- Learnable / LU (1×1) permutations.
- cond-concat hybrid conditioning.
- IAF / coupling transforms (fast high-dim NPE sampling).

## 2. Key architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Integration | **Parallel track** | The flow does not fit `GenerativeMethod` (no path/scheduler/solver/time/divergence). The flow *is* the density model. |
| Reuse | `AbstractPipeline.train()`/EMA/checkpointing + `diagnostics/` | These are method-agnostic. No `ConditionalWrapper` (the flow isn't a velocity field). |
| Tensor contract | **Pure `(batch, dim)`** core + `ch` adapter at the pipeline boundary | Matches flowjax/NumPyro and keeps the autoregressive masking math clean. SBI tabular data has `ch=1`. |
| Direction-agnostic core | One conditional flow `q(a\|b)`; NPE/NLE by relabeling | Same MADE/bijections/training; only the inference wrapper differs. |
| Conditioning | **FiLM + gate, no norm** (AdaLN-Zero mechanism minus cross-feature norm) | More expressive than concat; rank-safe; reuses the flux1/pixeldit modulation idiom. See §6. |
| NLE ↔ NumPyro | **Potential function** + NUTS, *not* a `Distribution` wrapper | NNX modules and NumPyro transforms fight each other (handout §10). |

### Why a discrete flow can't reuse `GenerativeMethod`
`GenerativeMethod` is CNF-shaped: `build_path`, `prepare_batch → (x_0, x_1, t)`,
`build_solver` (ODE/SDE), `sample_init` (prior noise), `build_log_prob_fn`
(continuous change-of-variables + divergence trace). A discrete MAF has none of
these: direct max-likelihood loss, exact one-pass `log_prob`, one inverse-pass
sampling. Forcing it behind that interface would be a leaky fit. Hence a parallel
track that still reuses the genuinely shared machinery (training loop, EMA,
checkpointing, diagnostics).

## 3. Module layout

New package `gensbi/normalizing_flows/` (named to avoid collision with the
existing `flow_matching/`, which is a *CNF* method):

```
src/gensbi/normalizing_flows/
  __init__.py
  bijections/
    base.py          # Bijection ABC: forward / inverse (each returns (out, log_det))
    masked_linear.py # MaskedLinear (NNX): dense + non-Param Mask buffer
    made.py          # MADE conditioner (NNX) + MaskedAutoregressive bijection ("MAFLayer")
    transformers.py  # elementwise maps: Affine (v1), RQSpline (v2) — pure param-driven math
    permutation.py   # Permutation bijection (reverse / alternating / fixed-random)
    chain.py         # Chain of bijections
  flow.py            # Flow (NNX): base_dist + Chain; log_prob / sample; make_maf builder

src/gensbi/recipes/flow_pipeline.py   # ConditionalFlowPipeline(AbstractPipeline) — Layer B (NPE)
src/gensbi/inference/__init__.py      # new package
src/gensbi/inference/nle.py           # NLEPosterior — Layer C (NUTS wrapper)
```

## 4. Direction convention (locked)

- `forward`: **noise → data** (sampling). MAF: slow, sequential (`lax.scan`).
- `inverse`: **data → noise** (density). MAF: fast, one pass.
- `log_prob(x, cond) = base.log_prob(u) + log_det` where `u, log_det = chain.inverse(x, cond)`.

Every bijection method returns `(output, log_det)` where `log_det` is the
log-abs-det of *that method's* Jacobian. Affine: `inverse` logdet `= −a`,
`forward` logdet `= +a` (with `a` the log-scale). Sign-for-sign per handout §3.

## 5. Layer A — flow core

### Bijection contract (`bijections/base.py`)
```python
forward(u, cond=None) -> (x, logdet)   # noise -> data
inverse(x, cond=None) -> (u, logdet)   # data  -> noise
```
Structural pieces (`MaskedLinear`, `MaskedAutoregressive`, `Permutation`,
`Chain`, `Flow`) are **NNX modules**. Elementwise **transformers** (`Affine`,
`RQSpline`) are **pure param-driven math** (not modules); they receive params
from MADE and expose `forward(u, params)`, `inverse(x, params)`, `num_params`.

### Components
- **`MaskedLinear`** — dense layer; mask held as a dedicated `Mask(nnx.Variable)`
  (non-`Param` buffer) so `nnx.split(wrt=Param)` and the optimizer never touch
  it, while checkpointing still saves/restores it.
- **`MADE`** — conditioner. **cond enters only via modulation, not the masked
  input** (the FiLM choice). Masked input layer sees `x` (ranks `1..D`); a small
  unmasked MLP embeds `cond`; each hidden block is the rank-safe modulated
  residual block (§6). Output `MaskedLinear → (D, num_params)` using the
  **strict `<`** mask, tiled across the `num_params` groups. One MADE emits all
  params (`out_ranks = repeat(arange(D), num_params)`, flowjax-style).
- **`MaskedAutoregressive`** (handout's "MAFLayer", MADE + transformer) —
  `inverse` = one MADE pass + vmapped `transformer.inverse` (fast);
  `forward` = `jax.lax.scan` over dims (sequential).
- **`Affine` transformer (v1)** — `num_params=2` (shift μ, log-scale `a`); `a`
  clamped to `[-5, 3]` via the stop-grad clamp trick (`x + stop_grad(clip(x)-x)`,
  NumPyro IAF). `inverse`: `u=(x−μ)·e^{−a}`, logdet `=−a`. `forward`:
  `x=u·e^{a}+μ`, logdet `=+a`.
- **`RQSpline` transformer (v2)** — `num_params = 3K−1` for `K` bins; monotonic
  rational-quadratic spline on `[−B, B]`, linear tails; `softmax` widths/heights
  (positive, sum to range), `softplus` internal derivatives (monotonic); analytic
  inverse; logdet `=` sum of log spline-derivatives. Only change vs v1: a new
  transformer subclass + wider MADE output.
- **`Permutation`** — index buffer (non-`Param`); reverse / alternating /
  fixed-random; logdet `= 0`.
- **`Chain`** — bijections in noise→data order; `inverse` walks them in reverse,
  accumulating logdet.
- **`Flow`** — `base_dist` (`core.prior.make_gaussian_prior`) + `Chain`.
  `log_prob(x, cond)`; `sample(key, cond, shape)`. **Standardization is a fixed
  `Affine` bijection at the data end of the chain** (mean/std as non-`Param`
  buffers set from training data) → the Jacobian correction is automatic; no
  special-casing in `log_prob`.
- **Builder** `make_maf(rngs, dim, cond_dim, n_layers, transformer="affine",
  nn_width, nn_depth, permutation="reverse", **transformer_kwargs)`.

## 6. Conditioning — concatenation at rank −1 (flowjax-style)

> **Decision (2026-06-21, supersedes the FiLM design below).** Phase 0 conditions
> the MADE by **concatenating** `cond` onto the input and assigning it
> autoregressive **rank −1** (below every data dim), as in the reference `flowjax`
> (`reference/flowjax/.../masked_autoregressive.py`). With the strict output mask
> (`out_rank > in_rank`), every output dim — including `d=0` — may depend on `cond`,
> while `cond` depends on nothing. This is the standard conditional-MAF approach
> (Papamakarios et al. 2017). The FiLM/adaLN-Zero scheme described below routes
> `cond` only through the hidden stream, which leaves `d=0` (whose output reads no
> hidden unit) **unconditioned** — a real bug for conditional density estimation.
> The FiLM text is retained as a candidate *alternative* conditioner (alongside
> e.g. T-NAF) for a future extension, behind the same `(x, cond) -> params`
> interface; it is not used in Phase 0. The "no cross-feature normalization"
> caveat still applies to any conditioner.

### (superseded) FiLM + gate variant

GenSBI already uses AdaLN-Zero modulation (`models/flux1/layers.py:Modulation`;
`experimental/models/pixeldit/blocks.py`: `x = x + gate * mlp(norm(x)*(1+scale)+shift)`
with zero-init adaLN). We adopt the **modulation mechanism** but **not the
cross-feature normalization**:

```python
# h carries autoregressive rank (via masked linears); c = embed(cond), cond-only
shift, scale, gate = adaLN(c)                          # cond-only, gate zero-init
h = h + gate * MaskedMLP( h * (1 + scale) + shift )    # per-unit mod + same-rank residual
```

**Why no LayerNorm/RMSNorm/GroupNorm:** MADE hidden units *carry the
autoregressive rank* — that is the mechanism. Cross-feature norms compute
statistics over all hidden units (including higher-rank ones) and feed them back
into every unit, injecting high-rank-input dependence into low-rank outputs. That
**silently breaks the autoregressive property** — `log_prob` stops being a valid
density and *nothing crashes*. `scale`, `shift`, `gate` are cond-only and
per-unit, so they are rank-safe; the zero-init gate gives each block an
identity warm-start (the actual source of AdaLN-Zero's stability — not the norm).
If a normalization is ever wanted, the only rank-safe kind is a **per-unit**
ActNorm (data-init then fixed per-unit scale/bias, no cross-unit statistics).

This holds for both directions: cond is the non-autoregressive variable in both
NPE (cond = x) and NLE (cond = θ).

## 7. Layer B — `ConditionalFlowPipeline(AbstractPipeline)`

Direction-agnostic: trains `q(obs | cond)` by max-likelihood.

- **NPE:** `obs = θ`, `cond = x` — same `(obs, cond)` convention as the existing
  `ConditionalPipeline`, so it is a drop-in for comparison.
- **NLE:** `obs = x`, `cond = θ`.

Reuses `AbstractPipeline.train()` / EMA / checkpointing / optimizer unchanged.
Overrides only:
- `_wrap_model` → identity (no `ConditionalWrapper`; the flow *is* the model).
  `ema_model` still works — param averaging over `nnx.Param`; masks/buffers are
  non-`Param` so they're untouched.
- `get_loss_fn` → `loss_fn(model, batch, key)`: `obs, cond = batch`; **adapter
  squeezes the `ch` axis** (`assert ch == 1`) → `(B, dim)`; return
  `−mean(flow.log_prob(obs, cond))`. `key` unused.
- `get_sampler` / `sample` → `flow.sample(key, cond=x_o)`; adapter re-expands
  `ch` → `(n, dim_obs, ch)`.
- `get_log_prob_fn` / `log_prob` → `flow.log_prob(obs, cond)`.
- `fit_standardization(data)` → set the standardize-bijection buffers from
  training stats.

For **NPE this pipeline is already the posterior**: `.sample(x_o, n)` /
`.log_prob(θ, x_o)` mirror `ConditionalPipeline`, so SBC / TARP / C²ST in
`diagnostics/` run unchanged → clean **FM-NPE vs NF-NPE** comparison.

### MAF sampling caveat (NPE)
NPE samples the autoregressive target (θ), which for MAF is the *slow*
sequential direction (D passes for D dims). For SBI, θ is usually low-dim
(2–20), so this is cheap. If θ is ever high-dim, that is the motivation for an
IAF/coupling transformer variant (deferred).

## 8. Layer C — `NLEPosterior` (`inference/nle.py`)

Takes an NLE-trained flow (`obs = x`, `cond = θ`) + a NumPyro prior over θ + an
observation `x_o`. Builds `potential(θ) = −[flow.log_prob(x_o, cond=θ) +
prior.log_prob(θ)]` and hands it to **NumPyro NUTS** (potential-function route,
not a `Distribution` wrapper). `∇_θ log q` is free via autodiff (the flow is
differentiable in its conditioning input). Exposes `.sample(key, x_o, n) ->
(n, dim_θ, ch)` so the **same diagnostics** run for NLE too. Amortized: one
trained flow serves any `x_o`.

## 9. Numerical-stability checklist
- Clamp affine log-scale to `[−5, 3]` (stop-grad clamp).
- Spline: positive widths/heights summing to range; positive (softplus)
  derivatives; correct `±B` tail handling.
- Standardize inputs via the fixed `Affine` standardize-bijection (Jacobian
  handled automatically by the chain).
- log-space everything (`log_prob` / log-sum-exp); never exponentiate densities.
- Masks as non-`Param` buffers so `nnx.split(wrt=Param)` / the optimizer never
  sweep them in.

## 10. Phasing (one design, staged build)
- **Phase 0 — core:** `base` / `MaskedLinear` / `MADE` (+FiLM-gate) / `Affine` /
  `Permutation` / `Chain` / `Flow`. Full official battery (§11). ⇒ correct affine
  MAF density.
- **Phase 1 — NPE:** `ConditionalFlowPipeline` + standardization. ⇒ NF-NPE,
  comparable to FM-NPE via existing diagnostics.
- **Phase 2 — NLE:** `NLEPosterior` (NUTS). ⇒ NF-NLE.
- **Phase 3 — RQ-NSF:** `RQSpline` transformer + wider MADE output; re-run the
  battery with the spline. ⇒ more expressive, both methods.

## 11. Testing

**Official battery (CI) — cheap, deterministic, fast unit checks:**
1. **MADE-Jacobian** — autodiff the MADE/flow `inverse`; assert output `d` has
   zero gradient w.r.t. `x_{≥d}` (autoregression) and dense gradient w.r.t.
   `cond` — **with modulation active** (guards the FiLM rank-safety claim).
2. **Invertibility** — `forward∘inverse ≈ id` and `inverse∘forward ≈ id`, tight
   tolerance.
3. **Log-det vs autodiff** — analytic logdet vs
   `jnp.linalg.slogdet(jax.jacobian(inverse))` on small `D` (~5 lines; the test
   that catches sign/convention bugs).
4. **Density integrates to 1 — 1D only** — `jnp.trapz(exp(log_prob))` over a 1D
   grid (cheap; "better than nothing").
5. **mask-is-buffer** — masks `∉ nnx.state(model, nnx.Param)`; optimizer never
   updates them. Plus clamp/stability sanity.

**Exploratory (run by hand, not in CI):**
- End-to-end linear-Gaussian (analytic posterior): NPE and NLE both recovered,
  cross-checked with existing SBC/TARP. Run manually during development.

**Deferred to an example notebook (later):**
- Two-moons / GMM unconditional fit (qualitative eyeball, not an assertion).

## 12. References
- Papamakarios, Pavlakou & Murray (2017) — *Masked Autoregressive Flow*.
- Germain, Gregor, Murray & Larochelle (2015) — *MADE*.
- Papamakarios, Sterratt & Murray (2019) — *Sequential Neural Likelihood*.
- Durkan, Bekasov, Murray & Papamakarios (2019) — *Neural Spline Flows*.
- Peebles & Xie (2023) — *DiT* (AdaLN-Zero modulation).
- Reference impls: `reference/flowjax` (algorithm); NumPyro `distributions/flows.py`
  (IAF clamp trick); GenSBI `models/flux1` & `experimental/models/pixeldit`
  (modulation idiom).

## 13. Decisions log
- Scope: full A+B+C design, phased build.
- Architecture: parallel track; reuse train loop / EMA / checkpointing /
  diagnostics; not `GenerativeMethod`.
- Tensor contract: pure `(batch, dim)` core + `ch` adapter.
- Methods: general conditional core; NPE + NLE inference now; amortized only.
- Conditioning: FiLM + gate, no norm (rank-safe AdaLN-Zero mechanism).
- Defaults: one MADE all params; reverse permutation (+alternating option); masks
  as non-`Param` buffers; NLE via NumPyro potential + NUTS; base via
  `core.prior.make_gaussian_prior`; standardization as a fixed affine bijection.
- Testing: official battery = MADE-Jacobian, invertibility, log-det, 1D density
  (trapz), mask-is-buffer; end-to-end linear-Gaussian = exploratory (not CI);
  two-moons = deferred example.
</content>
</invoke>
