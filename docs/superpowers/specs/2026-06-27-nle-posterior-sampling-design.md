# NLE Posterior Sampling: `gensbi.inference` + blackjax samplers

**Date:** 2026-06-27
**Status:** Design approved, ready for implementation plan
**Branch:** `maf` (un-merged, not pushed — breaking changes allowed; `NLEPosterior` is not in `main`)

## Context

NLE posterior sampling currently lives in `src/gensbi/inference/nle.py` as a single
75-line `NLEPosterior` class. It builds a potential
`U(θ) = −(log q(x_o|θ) + log p(θ))` from a trained likelihood flow plus a numpyro
prior, and runs **numpyro NUTS** through the potential-function route. It is consumed
by 4 test files and 3 recovery scripts, and returns samples shaped `(n, dim_θ, 1)`.

Two things motivate this work:

1. **A lone one-file `inference/` package feels wrong.** The fix is not to relocate it
   but to *populate* it: NLE inference is a distinct architectural layer that *consumes*
   trained models — a direct sibling of `diagnostics/` (which consumes trained
   models to evaluate posteriors). It earns its own top-level package.
2. **Swap the sampler from numpyro to blackjax.** Default to Microcanonical Langevin
   Monte Carlo (MCLMC); add tempered Sequential Monte Carlo (SMC) as an option for
   multimodal posteriors. blackjax ≥1.5 is already a declared dependency, used nowhere
   yet, and ships everything required (`mclmc`/`adjusted_mclmc` + tuners,
   `adaptive_tempered_smc`, NUTS/HMC inner kernels). Nested sampling is deferred until
   blackjax merges its pending PR; it will slot in later as one more sampler.

### Design philosophy (scope guard)

This is a **simple, easy-to-use NLE posterior sampler that reasonably works out of the
box** — a dependable one-liner with sensible defaults and a lean knob count. The NLE
power user who wants bespoke chains already has their `prior` and their neural
likelihood (`flow.log_prob`) and can drive blackjax or numpyro directly; we do not try
to be an exhaustive, maximally-tunable MCMC toolkit. This philosophy justifies the v1
cuts below (fixed inner-kernel parameters, no per-temperature tuning) and keeps the
constructors tight rather than exposing every blackjax dial.

## Goals

- Populate a top-level `gensbi.inference` package: `posterior.py` (the `NLEPosterior`
  *target* builder) + `samplers.py` (the sampler classes).
- Cleanly separate the **posterior target** (NLE-specific glue: `(flow, prior, x_o)` →
  log-densities) from the **sampler** (generic: log-density → samples).
- Default sampler = **adjusted MCLMC** (asymptotically exact, microcanonical).
- Optional **tempered SMC** for multimodal posteriors, with a microcanonical
  (adjusted-MCLMC) inner kernel by default.
- Keep numpyro as the prior abstraction; remove only the numpyro *sampler* usage.
- Preserve the `(n, dim_θ, 1)` output contract so existing tests/scripts migrate with
  minimal churn.

## Non-goals (explicit follow-ups)

- **Nested sampling.** Deferred until supported in blackjax; will be a new `Sampler`
  subclass with zero changes to `NLEPosterior`.
- **Per-temperature inner-kernel tuning inside SMC.** v1 uses fixed, configurable inner
  kernel parameters. Wiring `adjusted_mclmc_find_L_and_step_size` per temperature is a
  later refinement.
- **An unadjusted-MCLMC inner kernel for SMC.** Theoretically inappropriate (not
  MH-invariant) and API-incompatible with blackjax's SMC inner-kernel slot.
- **Migrating the prior abstraction off numpyro.** `make_gaussian_prior` and the
  numpyro-based prior are used across `core`/`recipes` and stay as-is.
- **Backward-compatibility shims.** `NLEPosterior` is not in `main`; the constructor is
  restructured freely and the ~7 internal call sites are updated.

## Decision 1 — Module placement: top-level `gensbi.inference`, populated

```
src/gensbi/inference/
  __init__.py        # exports: NLEPosterior, MCLMC, TemperedSMC, Sampler
  posterior.py       # NLEPosterior — builds the posterior target, dispatches to a sampler
  samplers.py        # Sampler (ABC), MCLMC (default), TemperedSMC
```

Rationale: `core/` has a narrow meaning in this repo — the `GenerativeMethod` strategy
and its flow-matching/diffusion/score-matching implementations (define a generative
process, train it, sample from it via its *own* solver). NLE posterior sampling is a
downstream operation on an *already-trained* model: combine the learned likelihood with
a prior and run an *external* inference algorithm to draw from the **posterior** — a
distribution the model was never trained to sample. That is the same layer as
`diagnostics/` (top-level, consumes trained models), so `inference/` is its peer, not a
`core` feature. Keeping it top-level also confines the `blackjax` dependency to the one
layer that needs it, leaving `core` lean.

## Decision 2 — The target ↔ sampler seam

`NLEPosterior` becomes "build a posterior **target** for a given `x_o`," not "run a
sampler." A target is a small frozen bundle:

```python
@dataclass(frozen=True)
class PosteriorTarget:
    log_prior:      Callable   # log p(θ)
    log_likelihood: Callable   # log q(x_o | θ)   — split out; SMC tempers this term
    log_posterior:  Callable   # log_prior + log_likelihood
    prior:          object     # numpyro dist — particle/chain init via .sample
    dim:            int
```

Samplers consume only the target:

```python
class Sampler(ABC):
    @abstractmethod
    def run(self, key, target: PosteriorTarget) -> tuple[Array, object]:
        """Returns (samples (n, dim), info)."""
```

Dispatch:

```python
class NLEPosterior:
    def __init__(self, flow, prior, *, structured_obs=False): ...

    def sample(self, key, x_o, sampler=None, *, return_info=False):
        sampler = sampler or MCLMC()              # adjusted MCLMC is the default
        target  = self._build_target(x_o)         # absorbs all structured_obs reshaping
        samples, info = sampler.run(key, target)  # sampler sees only flat θ
        samples = _expand_dims(samples)           # (n, dim, 1) — unchanged contract
        return (samples, info) if return_info else samples
```

Why this seam:

- **`structured_obs` is absorbed in `_build_target`** (where `x_o` is reshaped before
  `flow.log_prob`); θ is always a flat `(dim,)` vector, even in the field-NLE path, so
  samplers stay observation-agnostic.
- **Splitting `log_prior` and `log_likelihood`** is mandatory for tempered SMC (it
  tempers `p(θ)·q(x_o|θ)^β`) and free for MCLMC (which uses `log_posterior`).
- **Nested sampling later = one new `Sampler` subclass**, no change to `NLEPosterior`.

The sampler is selected per `sample()` call (default MCLMC one-liner; pass a configured
object to switch), so one `NLEPosterior` can be reused across samplers.

## Decision 3 — `MCLMC` sampler (default), adjusted by default

Microcanonical Langevin Monte Carlo. `run()` has three internal stages:

1. **init** — start position from `target.prior.sample`.
2. **tune** — auto-find trajectory length `L`, `step_size`, and a diagonal mass matrix.
3. **sample** — run a `lax.scan` loop for `num_samples`; `num_chains > 1` → `vmap` over
   split keys, reshaped to `(num_chains·num_samples, dim)`.

```python
MCLMC(*, adjusted=True, num_samples=1000, num_tuning_steps=5000, num_chains=1,
      diagonal_preconditioning=True)
```

**`adjusted` flag (default `True`).** Both variants integrate the same microcanonical
(isokinetic) dynamics, which leave the posterior invariant only in continuous time;
finite step size `ε` introduces a discretization bias that shrinks only as `ε→0`.

- **Adjusted** (`blackjax.adjusted_mclmc` + `adjusted_mclmc_find_L_and_step_size`) adds a
  Metropolis–Hastings accept/reject → **asymptotically exact / unbiased**, like NUTS,
  while keeping most of MCLMC's efficiency. Tuned to a target acceptance rate.
- **Unadjusted** (`blackjax.mclmc` + `mclmc_find_L_and_step_size`) omits the correction →
  faster, but biased (bias bounded by the energy-variance target, mostly affecting tails
  and marginal scale).

Default is **adjusted** because (a) θ in SBI is low-to-moderate dimensional, the regime
where unadjusted's speed edge is smallest while its bias liability is unchanged; and
(b) GenSBI ships calibration diagnostics (SBC, TARP, coverage) whose validity assumes an
exact sampler — a biased default would confound "flow is wrong" with "sampler is
biased." `adjusted=False` opts into the faster biased variant for high-dim throughput.

*Implementation note:* during implementation, confirm the adjusted tuner converges
cleanly on the analytic Gaussian test posterior. If it proves fragile as a default
one-liner, fall back to `adjusted=False` as the default and document that rigorous
calibration runs should set `adjusted=True` (or use the exact NUTS-based SMC path). The
flag and both code paths ship regardless; only the default may change.

## Decision 4 — `TemperedSMC` sampler (multimodal option)

`blackjax.adaptive_tempered_smc` walks particles along `p(θ)·q(x_o|θ)^β` for `β: 0→1`.
This is the multimodal escape hatch.

```python
TemperedSMC(*, num_particles=1000, target_ess=0.5, num_mcmc_steps=10,
            inner_kernel="mclmc",            # "mclmc" (adjusted) | "nuts"
            inner_step_size=0.1, inner_num_integration_steps=5,
            inner_inverse_mass_matrix=None)  # None -> ones(dim)
```

- **Inner (rejuvenation) kernel defaults to adjusted MCLMC** — the microcanonical
  formulation, consistent with Decision 3. It is the variant that belongs inside SMC:
  it is MH-invariant (what the mutation step needs) and API-compatible with blackjax's
  inner-kernel slot (`init(position, logdensity_fn) -> HMCState`, built on the HMC state
  machinery). **NUTS is the fallback** (`inner_kernel="nuts"`): the most-trodden,
  zero-trajectory-tuning path.
- **Adaptive tempering** holds `target_ess=0.5` to choose the `β`-ladder automatically —
  no hand-built schedule.
- Particles initialized from the prior; **systematic resampling**; runs to `β=1`.
- **v1 keeps inner-kernel `step_size`/`num_integration_steps` fixed and configurable**
  (not re-tuned per temperature). SMC's reweight+resample is forgiving of a
  roughly-set inner kernel; per-temperature tuning is a later refinement (non-goal).

Output is the final `(num_particles, dim)` particle cloud (so `n == num_particles`).

## Decision 5 — Output contract, `info`, error handling

- **Output unchanged:** `sample()` returns `(n, dim, 1)` via `_expand_dims`.
  `n = num_samples·num_chains` (MCLMC) or `num_particles` (SMC). Existing shape
  assertions in the 4 tests + 3 scripts continue to hold.
- **`return_info=False` default** → bare samples. `return_info=True` → `(samples, info)`,
  a small per-sampler dataclass:
  - `MCLMC`: tuned `L`, `step_size`, acceptance rate, `num_samples`, `num_chains`,
    divergence count.
  - `TemperedSMC`: **`log_evidence` (log Z)**, `num_temperature_steps`, final `β`, inner
    acceptance, divergence count.
- **No clamping of `log_prob`.** Stays faithful to the existing no-clamp decision
  (untrained flows can overflow in float32 — that is the model's truth, not the
  sampler's to mask). Samplers instead surface blackjax **divergence counts** in `info`
  so a bad run is *detectable* rather than silently NaN.
- `_build_target` validates `dim` against `prior.event_shape` and raises a clear error on
  mismatch.
- `blackjax` imported lazily inside `run()` so importing the package stays cheap and
  CPU-safe.

## Decision 6 — Testing (TDD)

- **Correctness anchor (both samplers).** Reuse the analytic Gaussian recovery from
  `test_nle.py`: `GaussianMock` likelihood `N(x;θ,I)` + prior `N(0,I)` ⇒ posterior
  `N(x_o/2, ½I)`; assert mean ≈ `x_o/2`, var ≈ `0.5`. Plus prior-recovery (zero-init
  flow ⇒ posterior = prior).
- **Target unit tests.** `log_posterior == log_prior + log_likelihood`; value and grad
  finite (the existing potential-equivalence test, sign-flipped to log-density).
- **The SMC payoff test (new).** A bimodal mock likelihood (log-sum-exp of two Gaussians
  at ±μ) under a broad prior where a single MCLMC chain stays stuck in one mode but
  `TemperedSMC` populates *both*; assert both clusters are recovered. This verifies the
  multimodal claim rather than assuming it.
- **Coverage.** `adjusted=True`/`False` both run; `return_info=True` exposes
  `log_evidence`; `structured_obs` smoke test (the field-NLE test, NUTS → MCLMC).
- **Speed.** Fast CPU tests stay small (low samples/particles/tuning), mirroring the
  current smoke style. The 3 recovery scripts (`maf_nle_recovery.py`,
  `tarflow_nle_recovery.py`, `tarflow_field_nle_recovery.py`) are updated to the new API
  and remain the GPU full-recovery gate run manually.

## Migration / blast radius

Internal only, on the un-merged `maf` branch:

- `inference/nle.py` → `inference/posterior.py` (+ new `inference/samplers.py`).
- `NLEPosterior` constructor: drop `num_warmup`/`num_samples`/`num_chains` (knobs move to
  sampler objects); keep `flow`, `prior`, `structured_obs`.
- Update ~7 call sites: `tests/normalizing_flows/test_nle.py`,
  `tests/models/tarflow/test_structured_boundary.py`,
  `tests/models/tarflow/test_structured_integration.py`,
  `tests/models/tarflow/test_pipeline_integration.py`, and the 3 recovery scripts.
- `pyproject.toml`: numpyro stays (prior abstraction); blackjax already declared.
