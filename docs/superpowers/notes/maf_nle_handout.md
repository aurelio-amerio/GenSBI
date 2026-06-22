# Masked Autoregressive Flow for NLE — Implementation Handout

*Kickstart document for the normalizing-flows feature brainstorm.*

---

## 1. Context & goal

We are extending an existing JAX SBI library (which already has **flow matching** and **diffusion** models) with **discrete normalizing flows**. Constraints and stack:

- **Networks:** Flax NNX
- **Distributions:** NumPyro (used for the base distribution only — see §10)
- **Primary target:** **Neural Likelihood Estimation (NLE)** — learn a conditional density `q(x | θ)` over data given parameters, then evaluate it fast inside MCMC.

**Plan:** implement **affine MAF first**, then **autoregressive RQ‑NSF** as a small diff on top of it (§8). Two flows, mostly one implementation.

---

## 2. Why MAF first (rationale recap)

The usual knock on MAF is slow sampling (its forward, noise→data, pass is sequential: D passes for D dims). **NLE never exercises that direction:**

- NLE evaluates `log q(x_o | θ)` at the **fixed observed** `x_o` for many proposed `θ` during MCMC. The flow is only ever *evaluated*, never *sampled*, in the inner loop.
- Density evaluation is MAF's **fast, one-pass** direction (all conditioner outputs computable at once because the full `x` is known).
- A discrete flow gives an **exact, differentiable** `log q` in a single autodiff pass → cheap `∇_θ log q(x_o | θ)` for HMC/NUTS. (This is the part flow matching makes expensive: likelihood from a CNF needs an ODE solve + divergence term, and the gradient differentiates through the solve.)
- Autoregressive transforms are **more expressive per layer** than coupling (each dim conditions on the *full* prefix, not half the vector) — and the coupling advantage (fast sampling) is irrelevant here.
- **Precedent:** Sequential Neural Likelihood (Papamakarios, Sterratt & Murray, 2019) used MAF as its density estimator. MAF and NLE grew up together.

---

## 3. The math to implement (MAF core)

Per dimension `i`, with conditioner outputs `(μ_i, α_i) = MADE(x_{1:i-1}; θ)`:

- **Inverse / density direction (data → noise), FAST, parallel:**
  `u_i = (x_i − μ_i) · exp(−α_i)`
- **Forward / sampling direction (noise → data), slow, sequential:**
  `x_i = u_i · exp(α_i) + μ_i`
- **Log abs det Jacobian** (data→noise): `Σ_i (−α_i)`
- **Density:** `log q(x | θ) = log p_base(u) + Σ_i (−α_i)`

> **Pin down the direction convention early and write it on the whiteboard.** Half of all flow bugs are a forward/inverse or a sign-of-log-det mismatch. Decide which direction is "forward" in code and be ruthless about it.

---

## 4. Component breakdown (NNX modules)

| Component | Responsibility | Notes |
|---|---|---|
| `MaskedLinear` | Dense layer with a fixed binary weight mask | Mask is a **buffer**, not a trainable param (§5, §12) |
| `MADE` | Autoregressive conditioner; outputs per-dim transform params; conditioned on `θ` | The genuinely fiddly piece |
| `Transform` (elementwise bijector) | `forward` / `inverse` / `log_det` for one elementwise map | Affine now, spline later — swappable |
| `Permutation` | Reorder dims between layers | Reverse / random / learnable |
| `MAFLayer` | MADE + Transform + Permutation; exposes `forward`/`inverse` + logdet | One flow step |
| `Flow` | Stack of layers + NumPyro base dist; `log_prob`, (`sample`) | The `TransformedDistribution` wrapper |

Target interface (sketch):

```
class Transform:        # elementwise bijector
    def forward(self, x, params) -> (y, logdet)
    def inverse(self, y, params) -> (x, logdet)

class MAFLayer(nnx.Module):
    def inverse(self, x, cond) -> (u, logdet)   # density direction, fast
    def forward(self, u, cond) -> (x, logdet)   # sampling, sequential

class Flow(nnx.Module):
    def log_prob(self, x, cond) -> jnp.ndarray  # the workhorse for NLE
    def sample(self, key, cond, shape)          # optional / slow
```

---

## 5. The MADE conditioner (the fiddly part)

MADE = a feed-forward net whose weight masks enforce that output dim `d` depends only on inputs `< d`, in a single pass.

- **Degree assignment:** input units get degrees `1..D`; each hidden unit gets a degree in `1..D-1`; output units inherit the degree of the dim they parameterize.
- **Mask rules:**
  - Hidden→hidden / input→hidden: connect if `deg_prev ≤ deg_curr`.
  - Last hidden→output: **strict** `deg_prev < deg_out` (this strictness is what makes it autoregressive — get it wrong and dim `d` sees itself).
- **Multiple params per dim:** affine needs 2 outputs/dim (`μ`, `α`); spline needs `3K−1`. **Tile** the output mask across the parameter groups so each group respects the same ordering.
- **Single pass** yields all conditionals `p(x_1), p(x_2|x_1), …` at once.

> This is the part to unit-test in isolation first (§13) before wiring anything else.

---

## 6. Conditioning on θ (NLE specifics)

We need `q(x | θ)`, autoregressive **in x**, freely dependent **on θ**.

- Feed `θ` into MADE as **extra inputs with no autoregressive masking** — i.e. columns that are always fully connected to every hidden unit.
- This keeps the x-autoregression (and thus the one-pass evaluation property) intact while letting the conditioner depend arbitrarily on `θ`.
- **Open question (§15):** simplest is concatenating `θ` to the input; alternatives are FiLM / an embedding injected into hidden layers. Start simple, revisit if capacity is the bottleneck.

---

## 7. Affine transform (v1)

- Params per dim: shift `μ`, log-scale `α`.
- **Clamp `α`** to e.g. `[-5, 3]` using the stop-gradient clamp trick (`x + stop_grad(clip(x) - x)`) — NumPyro's IAF does exactly this. Without it, training diverges early.
- log-det contribution per dim = `α` (data→noise: `−α`). Mind the sign vs §3.

---

## 8. NSF upgrade (v2) — the diff

Swap the affine map for a **monotonic rational-quadratic spline** (Durkan et al., 2019). Almost everything else is unchanged.

- **MADE output dim:** `2` → `3K − 1` per dim for `K` bins (widths, heights, derivatives).
- Spline acts on `[−B, B]`; **linear tails** outside the bound.
- **Analytic inverse** exists (so sampling remains possible if ever needed); `log_det` = sum of log spline-derivatives at the evaluation point.
- Parameterization hygiene: `softmax`/`softplus` widths & heights to be positive and sum to the bin range; `softplus` the internal derivatives to stay monotonic.
- **Unchanged:** MADE, permutations, `Flow` wrapper, training loop, NLE integration.

Net effect: the second "flow" is a new `Transform` subclass + a wider MADE output. That's the whole point of the modular split.

---

## 9. Permutations / alternating direction

Without reordering, the first dim is never conditioned and the last never conditions. Between layers:

- Options: **reverse** (cheapest), **fixed random**, or **learnable** (LU / 1×1 linear).
- Start with reverse or fixed random; add learnable later if needed.
- **TarFlow insight that transfers even at small scale:** alternating the autoregression *direction* between layers is a cheap, effective alternative to fixed permutations.

---

## 10. Base distribution & the NumPyro boundary

- Base dist: `dist.Normal(0,1).to_event(1)` or `dist.MultivariateNormal`. We only need `base.log_prob` and `base.sample` — arrays in, arrays out.
- **Do NOT** wrap the flow as a NumPyro `Transform` / `TransformedDistribution`. NumPyro transforms are flat, static-ish pytrees; NNX modules are stateful objects with reference semantics that you `split`/`merge`. Mixing them fights both systems.
- **Keep the flow as an NNX module.** For MCMC, expose `flow.log_prob(x_o, θ)` and use it directly in the potential function (§11) — no need to make the flow itself a NumPyro distribution.

---

## 11. Integration with NLE + MCMC

**Training (amortized):**

- Simulate pairs `(θ_n, x_n)`.
- Maximize `Σ_n log q(x_n | θ_n)` with NNX + optax.
- Standardize `x` and `θ` (§12).

**Inference (per observed `x_o`):**

- Potential: `−[ log q(x_o | θ) + log prior(θ) ]`.
- Hand the potential to NumPyro **NUTS** (or your sampler of choice).
- Requires `∇_θ log q(x_o | θ)` — available by autodiff since `log_prob` is differentiable in the conditioning input.

**Workflow choice (§15):** amortized-only vs **sequential** NLE rounds (SNL: retrain on simulations drawn near the current posterior). Sequential is more sample-efficient but adds orchestration.

---

## 12. Numerical-stability checklist

- [ ] Clamp affine log-scale (§7).
- [ ] Spline: positive widths/heights summing to range; positive (softplus) derivatives; correct tail handling at `±B`.
- [ ] **Standardize inputs** — external standardization and/or `ActNorm` layers. (BayesFlow does both; worth copying the instinct.)
- [ ] Use `log_prob` / log-sum-exp forms; never exponentiate densities directly.
- [ ] Keep masks as **non-trainable buffers** so `nnx.split` doesn't sweep them into the optimizer state.

---

## 13. Testing strategy

Build confidence bottom-up:

1. **MADE masking** — assert output `d` has zero Jacobian wrt inputs `≥ d` (autodiff a single MADE pass on small `D`).
2. **Invertibility** — `forward(inverse(x)) ≈ x` to tight tolerance.
3. **Log-det correctness** — compare analytic `log_det` against `jnp.linalg.slogdet(jax.jacobian(...))` on small `D`. This catches sign/convention bugs immediately.
4. **Density sanity** — numerically integrate `exp(log_prob)` over a 1–2D grid; should be ≈ 1.
5. **Unconditional fit** — fit a toy target (two moons, Gaussian mixture); eyeball samples/density.
6. **Conditional / NLE end-to-end** — pick a toy model with a *known* likelihood, run NLE + NUTS, and check the recovered posterior against ground truth (or against analytic/long-MCMC reference).

---

## 14. References

- Papamakarios, Pavlakou & Murray (2017) — *Masked Autoregressive Flow for Density Estimation*.
- Germain, Gregor, Murray & Larochelle (2015) — *MADE*.
- Papamakarios, Sterratt & Murray (2019) — *Sequential Neural Likelihood* (NLE with MAF).
- Durkan, Bekasov, Murray & Papamakarios (2019) — *Neural Spline Flows*.
- **Reference implementations:** FlowJAX (algorithm reference; Equinox) · Bijx (NNX idioms) · NumPyro `distributions/flows.py` (IAF / clamp trick).

---

## 15. Open design questions for the session

- **θ-conditioning injection:** concat to inputs vs FiLM / embedding into hidden layers?
- **Permutation type:** fixed reverse vs random vs learnable — and is alternating-direction enough?
- **NumPyro coupling:** plain potential-function `factor` vs a thin custom `Distribution` wrapper — what's cleanest for the rest of the library?
- **Amortized vs sequential** NLE — support rounds now or later?
- **Mask storage in NNX:** confirm the cleanest way to hold masks out of trainable state.
- **Standardization:** `ActNorm` layers vs external preprocessing vs both?
- **Conditioner factoring:** one MADE emitting all params vs separate heads; same question for spline params.
- **Shape conventions:** batch / event-shape handling to match the existing flow-matching & diffusion modules.
- **How much to share with the existing CNF code** — base distributions, standardization, training loop, diagnostics?

---

*Deliverable order for the session: settle §15, lock the direction convention (§3), then build MADE → affine MAF → tests → NSF diff.*
