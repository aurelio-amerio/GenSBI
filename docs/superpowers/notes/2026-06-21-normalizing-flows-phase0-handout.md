# Normalizing Flows — Phase 0 Outcomes Handout

*What was actually built in Phase 0 (flow core), the decisions made during the
build, and the concrete seams for planning Phase 1 (NPE), Phase 2 (NLE), and
Phase 3 (RQ-NSF).*

*Companion to the pre-build brainstorm `maf_nle_handout.md` (which describes what
was **planned**) and the design spec
`docs/superpowers/specs/2026-06-21-normalizing-flows-design.md`. This document is
the kickstart input for the next-phase implementation plans.*

---

## 0. Status

- **Phase 0 = COMPLETE** on branch `maf`. 10 commits (`f286444..9f35483`).
- **28 unit tests pass.** Final cross-cutting review verdict: **SHIP** (no
  critical/important findings; math verified independently — densities integrate
  to 1, roundtrips exact to ~1e-7, log-det signs consistent, vmap-over-nnx sound).
- Run tests with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/`
  (the institute GPUs are usually busy; the `JAX_PLATFORMS=cpu` prefix is required).
- Untracked and **not yet committed**: this handout + the spec/plan under
  `docs/superpowers/`, and `reference/flowjax/` (a clone used as the algorithm
  reference — should be `.gitignore`d, not committed).

---

## 1. What was delivered

A standalone, correct **conditional affine MAF** as Flax-NNX modules, with exact
`log_prob` and one-pass conditional `sample`. A **parallel track** to the existing
flow-matching/diffusion methods — the flow **is** the density model (not the
CNF-shaped `GenerativeMethod` ABC).

### File / module map (all under `src/gensbi/normalizing_flows/`)

| File | Public API | Role |
|---|---|---|
| `bijections/base.py` | `Bijection` (ABC), `Mask` (`nnx.Variable`) | direction convention + non-Param buffer type |
| `bijections/masks.py` | `make_mask(in_ranks, out_ranks, *, strict)` | rank-based binary mask, shape `(in, out)` |
| `bijections/masked_linear.py` | `MaskedLinear(in, out, mask, rngs, param_dtype=f32)` | dense layer w/ fixed mask buffer |
| `bijections/transformers.py` | `Affine(clamp_min=-5, clamp_max=3)` | elementwise transformer (pure math, not a Module) |
| `bijections/made.py` | `MADE(...)`, `MaskedAutoregressive(...)` | conditioner + one autoregressive flow step |
| `bijections/permutation.py` | `Permutation.reverse(dim)` / `.random(dim, rngs)` | dim reordering, logdet 0 |
| `bijections/standardize.py` | `Standardize(dim)` + `.set_stats(mean, std)` | fixed affine at the data end |
| `bijections/chain.py` | `Chain(bijections)` | compose bijections |
| `flow.py` | `Flow(chain, dim, cond_dim)`, `make_maf(...)` | base dist + chain; batch via `jax.vmap` |

### Key signatures (so the next plans call them correctly)

```python
make_maf(rngs, dim, cond_dim=0, n_layers=5, transformer=None,
         nn_width=64, nn_depth=2, permutation="reverse",
         standardize=True, zero_init=True) -> Flow

Flow.log_prob(x, cond=None) -> (batch,)          # = base.log_prob(u) + logdet,  u,logdet = chain.inverse(x, cond)
Flow.sample(key, cond=None, nsamples=None) -> (n, dim)   # n defaults to cond.shape[0]
Flow.chain                                       # the Chain (used directly in tests / NLE potential)

MaskedAutoregressive(dim, cond_dim, transformer, nn_width, nn_depth, rngs, zero_init=True)
MADE(dim, cond_dim, num_params, nn_width, nn_depth, rngs, zero_init=True, param_dtype=f32, activation=silu)
MADE.__call__(x, cond=None) -> (dim, num_params)

Affine().num_params == 2          # params per dim = [shift mu, log-scale a]
Affine.forward(u, params)/inverse(x, params) -> (val, logdet);  forward_dim(u_i, params_i) -> x_i (scalar, scan)
Standardize(dim).set_stats(mean, std)   # in-place; defaults to identity
```

All modules operate on a **single example** `(dim,)`; `Flow` `jax.vmap`s over the
batch. **Float32 everywhere** (exact-likelihood model; bf16 would wreck the
Jacobian/log-det precision).

---

## 2. Direction convention (LOCKED — do not regress)

| method | direction | MAF cost | used by |
|---|---|---|---|
| `inverse` | data → noise | **fast**, one MADE pass | `log_prob` (density) |
| `forward` | noise → data | slow, `lax.scan` over dims | `sample` |

Each bijection returns `(output, log_det)` where `log_det` is the log\|det\| of
**that method's** Jacobian. Affine: `inverse` logdet `= -sum(a)`, `forward` logdet
`= +sum(a)`. `Chain` applies `forward` in list order, `inverse` in reverse order;
**`Standardize` is appended last** ⇒ applied **first** in `inverse` (the data end).

> For **NLE** (the headline use case) the inner MCMC loop only ever *evaluates*
> `log_prob` at the fixed `x_o` — MAF's fast direction. Sampling slowness is
> irrelevant there. For **NPE**, sampling θ is the slow direction, but θ is
> usually low-dim (2–20) so it's cheap.

---

## 3. Decisions made DURING the build (deviations from the original spec/plan)

These are the non-obvious changes the next-phase plans must be aware of:

### 3a. Conditioning: FiLM → **flowjax-style concatenation at rank −1** (IMPORTANT)
The spec §6 originally specified **FiLM/adaLN-Zero** modulation (cond enters only
via per-unit scale/shift/gate on the hidden stream). **This is a real bug for
conditional density estimation:** MADE output dim `d=0` reads *no* hidden unit
(strict output mask needs `hidden_rank < 0`), so it's pure output bias — FiLM
never reaches it, leaving `q(x_0 | cond)` **independent of cond**.

**Fix (now implemented):** the reference `flowjax` approach — **concatenate `cond`
onto the MADE input** and give it autoregressive **rank −1** (below every data
dim); hidden ranks shift to `[-1, dim-2]`. The strict output mask (`out_rank >
in_rank`) then lets *every* output dim including `d=0` depend on cond, while cond
depends on nothing. Standard conditional-MAF (Papamakarios et al. 2017).
- Caught by `test_made_depends_on_cond_densely` (the test was right; the plan's
  impl was wrong). Spec §6 and the plan were updated to record the decision.
- **The FiLM and T-NAF schemes are explicitly retained as *future* alternative
  conditioners** behind the same `(x, cond) -> (dim, num_params)` MADE interface.
  A future phase can add them as a swappable conditioner. (User intent noted.)

### 3b. `zero_init` redefined → **identity warm-start**
With FiLM gone, `zero_init` no longer means "zero the adaLN gate." It now means:
**zero the MADE output layer** ⇒ all transform params start at 0 ⇒ Affine is the
identity ⇒ the whole flow starts as the standard normal. Default `True`
(production); tests pass `False` so the net is live. This makes the 1D
density-integrates-to-1 test robust and is a standard NF warm-start.

### 3c. `nnx.List` for module lists
A plain Python `list` of `nnx.Module`s cannot be assigned to a Module attribute in
flax 0.12.x (raises `ValueError`). Use `nnx.List([...])` (established pattern in
`pixeldit`/`simformer`). Used in `MADE.hidden_layers` and `Chain.bijections`.
Note: `make_maf` passes a plain list to `Chain(...)`, which wraps it internally —
do not double-wrap.

### 3d. `.value` accessor — deprecation, deferred sweep
The whole subpackage uses `var.value` on NNX Variables, which now emits a flax
`DeprecationWarning` (newer repo code, e.g. `pixeldit`, uses `.get_value()` /
`var[...]`). Kept `.value` for internal consistency across the subpackage. **Repo-
wide cleanup item** — migrate before a flax bump removes `.value`. ~14 sites.

---

## 4. Test coverage (the spec §11 official battery — all green)

1. **MADE-Jacobian / autoregression** (`test_made_is_autoregressive`): output `d`
   has zero Jacobian w.r.t. `x_{>=d}`; dense w.r.t. `x_{<d}`. Plus
   `test_made_depends_on_cond_densely`: every output (incl. `d=0`) depends on cond.
2. **Invertibility** (`MaskedAutoregressive`, `Chain`, `Flow`): `forward∘inverse ≈
   id` and back, tight tol.
3. **Log-det vs autodiff** (`MaskedAutoregressive`, `Chain`, `Flow`): analytic
   logdet vs `slogdet(jacobian(inverse))`.
4. **Density integrates to 1 (1D)** (`Flow`): trapezoid of `exp(log_prob)` ≈ 1.
5. **mask-is-buffer** (`MaskedLinear`, `Flow`): masks ∉ `nnx.state(model, Param)`.
Plus added hardening: zero-init identity warm-start, unconditional (`cond_dim=0`)
path, and the missing-cond `ValueError`.

---

## 5. Known issues / Phase-1 watch-items (from the final review)

These are **fine now but will matter when training starts** (Phase 1):

- **`Standardize.set_stats` mutates buffers in place** and there is **no guard**
  that stats were actually set before training. Under `nnx.jit`, stats must be set
  *before* tracing or via a proper state update, else the identity default
  silently trains. The Phase-1 pipeline's `fit_standardization(data)` must set
  these, and ordering must be enforced.
- **Affine log-scale clamp uses a straight-through gradient** (`stop_gradient`
  trick from NumPyro IAF). Once `a` saturates at `clamp_max=3`/`clamp_min=-5`, the
  gradient still flows as if unclamped → the raw pre-clamp `a` can drift far with
  no restoring gradient while `exp(a)` is silently pinned. Add a **diagnostic
  histogram of raw `params[..., 1]`** during Phase-1 training; not a code change.
- **`_base()` rebuilds `make_gaussian_prior((dim,))` each call** — deliberate (so
  it never enters nnx state); negligible cost, traced away under `nnx.jit`.
- **`nn_depth=0`** (input→output directly) is untested; harmless no-op loop.
- **`.value` deprecation sweep** (§3d).

---

## 6. Seams for the next phases (how Phase 0 plugs in)

Phase 0 is the density core. The next phases wrap it; **none require touching the
bijections** (except Phase 3, which adds one new transformer).

### Phase 1 — NPE (`ConditionalFlowPipeline`)
- Subclass `AbstractPipeline`; reuse `train()`/EMA/checkpointing/optimizer.
  Override: `_wrap_model` → identity (the flow IS the model — **no
  `ConditionalWrapper`**); `get_loss_fn` → `obs, cond = batch`; **ch-adapter
  squeezes the `ch` axis** (`assert ch == 1`) → `(B, dim)`; loss `=
  -mean(flow.log_prob(obs, cond))`; `get_sampler`/`sample` → `flow.sample(key,
  cond=x_o)` then re-expand `ch`; `get_log_prob_fn` → `flow.log_prob`.
- **NPE convention:** `obs = θ`, `cond = x` (mirrors the existing
  `ConditionalPipeline`, so SBC/TARP/C²ST in `diagnostics/` run unchanged → clean
  FM-NPE vs NF-NPE comparison).
- **`fit_standardization(data)`** → `Standardize.set_stats(mean, std)` from train
  stats (heed the §5 ordering caveat). EMA over `nnx.Param` only — masks/buffers
  are non-Param so they're untouched (verified).

### Phase 2 — NLE (`NLEPosterior`, `inference/nle.py`)
- `obs = x`, `cond = θ`. **Do NOT** wrap the flow as a NumPyro
  `TransformedDistribution`. Expose `flow.log_prob(x_o, θ)` and use it directly in
  a NUTS **potential**: `-[ log q(x_o | θ) + log prior(θ) ]`. `∇_θ` is free by
  autodiff (log_prob differentiable in `cond`).

### Phase 3 — RQ-NSF (`RQSpline` transformer)
- Add an `RQSpline` transformer alongside `Affine`, same interface
  (`.num_params`, `.forward`/`.inverse`/`.forward_dim`). MADE's
  `num_params=transformer.num_params` already generalizes — **wider MADE output is
  automatic.** Re-run the §11 battery with the spline. No conditioner changes.
- Stability checklist: positive widths/heights summing to range; softplus
  derivatives; correct linear-tail handling at `±B`.

---

## 7. References

- Papamakarios, Pavlakou & Murray (2017) — *Masked Autoregressive Flow*.
- Germain et al. (2015) — *MADE*.
- Papamakarios, Sterratt & Murray (2019) — *Sequential Neural Likelihood* (NLE+MAF).
- Durkan et al. (2019) — *Neural Spline Flows* (Phase 3).
- **Algorithm reference (in-repo):** `reference/flowjax/` — esp.
  `flowjax/bijections/masked_autoregressive.py` (the rank-−1 concat conditioning)
  and `flowjax/masks.py`.
