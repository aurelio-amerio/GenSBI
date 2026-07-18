# Design: Port MCLMC + TemperedSMC to the blackjax 1.6 API

**Date:** 2026-07-11
**Branch:** `nested-sampling` (continues on the same branch; this port is the merge blocker
for landing the branch on `main`)
**Status:** Approved

## Problem

The `nested-sampling` branch bumped the dependency floor to `blackjax>=1.6` (required by
`blackjax.nss`), and the mamba `gensbi` env now runs blackjax 1.6. Under 1.6, the
pre-existing samplers in `src/gensbi/inference/samplers.py` break: **10 test failures**
(7 in `tests/inference/test_mclmc.py`, 3 in `tests/inference/test_smc.py`). The branch
must not merge while its own declared floor breaks its two default samplers.

Root causes (from a source-level diff of the installed 1.5 vs 1.6 trees; the repo `.venv`
still holds 1.5, the mamba `gensbi` env holds 1.6):

- **A (5 tests, unadjusted MCLMC):** `blackjax.mclmc_find_L_and_step_size` now requires an
  explicit `logdensity_fn=` kwarg (raises `ValueError` otherwise), and
  `blackjax.mcmc.mclmc.build_kernel` no longer accepts `logdensity_fn`/`inverse_mass_matrix`
  (they moved to per-call kernel arguments).
- **B (2 tests, adjusted MCLMC):** `blackjax.adjusted_mclmc_find_L_and_step_size` gained a
  required positional `logdensity_fn`, and the kernel closure it drives has a new call
  signature — integration-step tunables arrive via an `integration_steps_params` tuple, not
  a baked-in scalar; `build_kernel` no longer takes `inverse_mass_matrix`.
- **C (2 tests, TemperedSMC mclmc inner kernel):** `blackjax.mcmc.adjusted_mclmc.build_kernel`
  no longer accepts `logdensity_fn`/`inverse_mass_matrix`; the static kernel's
  `num_integration_steps: int` became `integration_steps_params: tuple`.
- **D (1 test, TemperedSMC NUTS inner kernel):** *not* an API break. blackjax 1.6 fixed a
  sign bug in the adaptive-tempering ESS bisection target (`smc/ess.py`, upstream issue
  #914: `-delta * logdensity` → `delta * logdensity`). Same test + seed passes under 1.5
  and fails under 1.6 with posterior mean ≈ (0.03, 0.02) instead of (0.5, −0.5) — the
  samples look prior-like, suggesting the corrected solver anneals λ 0→1 too fast for the
  fixed inner-kernel settings to rejuvenate.

The 1.6 theme behind A–C: blackjax's kernels no longer statically bind
`logdensity_fn`/`inverse_mass_matrix` in `build_kernel`; those are per-call kernel
arguments now, and the `*_find_L_and_step_size` tuners take `logdensity_fn` explicitly.

The entire blackjax surface to port lives in `src/gensbi/inference/samplers.py` — nothing
else under `src/` calls blackjax, there is no `scripts/` directory in this repo, and the
`NestedSampler`/`nss` code is 1.6-native and out of scope.

## Decision

**Minimal mechanical port, no public API changes, plus a dedicated quality pass for the two
upstream behavior changes.** Alternatives rejected:

- *Version-compat shim (support 1.5 and 1.6):* the floor is already `>=1.6`; dual-path code
  is pure liability.
- *Broader refactor onto blackjax's new `build_sampling_algorithm` conventions:* the tuners
  still require kernel closures, so there is little to gain beyond what the minimal port
  already simplifies.

Constructor signatures, knob names, defaults-as-API, and the `Sampler` contract
(`run(key, target) -> (samples, info)`) are unchanged. Per the module's design philosophy
("simple sampler that reasonably works out of the box"), we do **not** expose new blackjax
1.6 knobs (e.g. `target_num_integration_steps`) — we accept upstream defaults.

## Code changes (all in `src/gensbi/inference/samplers.py`, ~40 lines, 3 functions)

### 1. `MCLMC._run_unadjusted` (currently lines ~191–210) — root cause A

```python
kernel = blackjax.mcmc.mclmc.build_kernel(integrator=isokinetic_mclachlan)
state, params, _ = blackjax.mclmc_find_L_and_step_size(
    mclmc_kernel=kernel, logdensity_fn=target.log_posterior,
    num_steps=self.num_tuning_steps, state=init_state,
    rng_key=tune_key, diagonal_preconditioning=self.diagonal_preconditioning)
```

- Build the kernel **once** — delete the `lambda inverse_mass_matrix: build_kernel(...)`
  closure; the 1.6 tuner threads `logdensity_fn`/`inverse_mass_matrix`/`L`/`step_size`
  through per-call itself.
- `blackjax.mcmc.mclmc.init(...)` and the top-level `blackjax.mclmc(...)` factory are
  unchanged — keep as-is.
- Risk: low; 1.6 kernel math is unchanged (renames only).

### 2. `MCLMC._run_adjusted` (currently lines ~212–248) — root cause B

```python
kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
    integration_steps_fn=lambda k, avg: jnp.ceil(
        jax.random.uniform(k) * _rescale(avg)),
    integrator=isokinetic_mclachlan,
)
state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
    mclmc_kernel=kernel, logdensity_fn=target.log_posterior,
    num_steps=self.num_tuning_steps, state=init_state,
    rng_key=tune_key, target=self.target_acceptance,
    diagonal_preconditioning=self.diagonal_preconditioning)
```

- `build_kernel` loses `inverse_mass_matrix`; the kernel is built **once** (no per-tuner-step
  closure rebuild). `integration_steps_fn` becomes `(rng_arg, avg) -> steps`: the tuner
  passes the running average positionally via `integration_steps_params=(avg,)`.
- `adjusted_mclmc_dynamic.init(...)` and the top-level `blackjax.adjusted_mclmc_dynamic(...)`
  factory (with the single-arg `integration_steps_fn` closed over the final
  `params.L / params.step_size`) are unchanged — keep as-is.
- `_rescale` is untouched. `_check_rescale_domain`'s `mu >= 1` guard stays: 1.6's tuner
  anchors `L / step_size` near `target_num_integration_steps = 2.0`, which satisfies it.
- **Upstream behavior change to absorb (not just signatures):** 1.6 rewrote the adjusted
  tuner (avg-preserving calibration; removed the 1.5 √dim `L`-reset that collapsed
  `L/step_size` before pass-2 dual averaging). Tuned `L`/`step_size` — and therefore
  acceptance behavior — shift systematically. See Quality pass.

### 3. `TemperedSMC._inner`, mclmc branch (currently lines ~319–333) — root cause C

```python
kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(integrator=isokinetic_mclachlan)

def step_fn(rng_key, state, logdensity_fn, step_size,
            num_integration_steps, inverse_mass_matrix):
    return kernel(rng_key, state, logdensity_fn, step_size,
                  integration_steps_params=(num_integration_steps,),
                  inverse_mass_matrix=inverse_mass_matrix)
```

- Kernel built **once outside** `step_fn` (also removes the current per-SMC-step
  `build_kernel` rebuild wart, matching the nuts branch's existing style).
- `mcmc_parameters` dict keys stay exactly as SMC expects to thread them into `step_fn`.
- The 1.6 static `adjusted_mclmc` kernel body is unchanged beyond the parameter-passing
  refactor — low risk.
- NUTS branch (`blackjax.nuts.build_kernel()` / `.init`): signatures unchanged, **no code
  change**. `TemperedSMC.run` call sites (`adaptive_tempered_smc`, `extend_params`,
  `smc.init/step`): signatures unchanged, **no code change**.

## Quality pass (the non-mechanical part)

### SMC adaptive tempering — root cause D (owner call: diagnose + retune, not test-side patching)

1. **Diagnose first:** instrument the λ schedule under 1.6 (number of tempering steps and
   the λ trajectory for the failing analytic NUTS recovery case) to confirm or refute the
   "anneals too fast to rejuvenate" hypothesis.
2. **Audit blackjax 1.6 SMC-side defaults** for changes alongside the #914 fix — diff
   `smc/adaptive_tempered.py`, `smc/ess.py`, `smc/solver.py` (root solver, `target_ess`
   conventions, solver iteration caps) between the installed 1.5 (`.venv`) and 1.6
   (`gensbi` env) trees.
3. **Retune GenSBI's `TemperedSMC` defaults** (`num_mcmc_steps`, `target_ess`, inner-kernel
   step params) until `test_smc_nuts_analytic_gaussian_recovery` **genuinely** passes —
   the acceptance bar is real posterior recovery (mean ≈ x_o/2 within existing tolerance),
   NOT a loosened tolerance around prior-like samples. If defaults change, their docstrings
   change in the same commit.
4. The two SMC-MCLMC recovery tests (blocked behind root cause C today) use the same
   adaptive-tempering machinery and get the same scrutiny once they compile.

### Adjusted-MCLMC numerics

Re-validate `test_adjusted_analytic_gaussian_recovery` and
`test_adjusted_reports_acceptance_rate` under the rewritten 1.6 tuner. Expectation changes
must be justified by the upstream tuner change (documented in the commit), not used to mask
a porting bug. Update any `samplers.py` docstring that describes 1.5 tuner behavior
(e.g. `_rescale` / mu heuristics tied to old `L/step_size` trajectories).

## Testing & success criteria

- Existing suites are the spec: **all 21 tests in `test_mclmc.py` + `test_smc.py` pass**,
  and the full `tests/inference/` suite is green (those 21 + 11 nested + 4 target = 36).
- No new test files. New assertions only if the diagnosis step warrants pinning schedule
  behavior; otherwise the recovery tests already encode the acceptance bar.
- Test command (CPU-only; node GPUs busy):
  `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/ -v`
  in the mamba `gensbi` env (NOT `.venv` — it still holds blackjax 1.5 and would
  false-pass).

## Out of scope

- `NestedSampler` / `blackjax.nss` code (1.6-native, shipped, review-confirmed).
- Any public API change, new knob, or version-compat shim.
- Non-`samplers.py` code (nothing else touches blackjax).
- Merging `nested-sampling` to `main` (user owns that gate; this port unblocks it).
