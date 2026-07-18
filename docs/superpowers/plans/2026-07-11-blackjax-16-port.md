# Blackjax 1.6 Port (MCLMC + TemperedSMC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MCLMC` (unadjusted + adjusted) and `TemperedSMC` work correctly under blackjax 1.6, fixing all 10 failing tests in `tests/inference/` with zero public-API changes.

**Architecture:** Three mechanical call-site ports in `src/gensbi/inference/samplers.py` (blackjax 1.6 moved `logdensity_fn`/`inverse_mass_matrix` out of `build_kernel` into per-call kernel arguments and explicit tuner kwargs), followed by a diagnose-and-retune pass for the one upstream *behavior* change (the #914 adaptive-tempering ESS sign fix). Spec: `docs/superpowers/specs/2026-07-11-blackjax-16-port-design.md`.

**Tech Stack:** JAX, blackjax 1.6 (mamba `gensbi` env), pytest. The repo `.venv` still holds blackjax 1.5 — used ONLY for behavioral comparison in Task 4, never as a test gate.

## Global Constraints

- Work directly on branch `nested-sampling` (user decision; this port is the branch's merge blocker).
- Test command: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest <paths> -v` from the repo root. The mamba `gensbi` env has blackjax 1.6; `.venv` has 1.5 and would false-pass — never use it for the gate. The `JAX_PLATFORMS=cpu` shell prefix is required (node GPUs busy).
- Zero public API changes: constructor signatures, knob names, the `Sampler` contract (`run(key, target) -> (samples, info)`), and `MclmcInfo`/`SmcInfo` fields are all unchanged. Do NOT expose new blackjax 1.6 knobs (e.g. `target_num_integration_steps` — accept upstream's 2.0).
- The existing test files are the spec. Do NOT edit `tests/inference/test_mclmc.py` or `tests/inference/test_smc.py` unless a task step explicitly authorizes it; if a test seems to need changing, STOP and escalate instead.
- Any class default changed by Task 4 gets its docstring updated in the same commit.
- API facts (verified against installed blackjax 1.6 by smoke run on 2026-07-11 — trust these, do not re-derive):
  - `blackjax.mcmc.mclmc.build_kernel(integrator=...)` → kernel; `blackjax.mclmc_find_L_and_step_size(mclmc_kernel=kernel, logdensity_fn=..., num_steps=..., state=..., rng_key=..., diagonal_preconditioning=...)` works and returns `(state, params, ...)` with `params.L/.step_size/.inverse_mass_matrix`.
  - `blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(integration_steps_fn=lambda k, avg: ..., integrator=...)` → kernel; `blackjax.adjusted_mclmc_find_L_and_step_size(mclmc_kernel=kernel, logdensity_fn=..., num_steps=..., state=..., rng_key=..., target=..., diagonal_preconditioning=...)` works. The tuner passes the running average via `integration_steps_params=(avg,)`, which arrives as `integration_steps_fn`'s second positional argument. Tuned `params.L / params.step_size` anchors at exactly **2.0** (new upstream `target_num_integration_steps` default) — passes `_check_rescale_domain` (≥ 1).
  - `blackjax.mcmc.adjusted_mclmc.build_kernel(integrator=...)` → kernel callable as `kernel(rng_key, state, logdensity_fn, step_size, integration_steps_params=(n,), inverse_mass_matrix=imm)`.
  - The top-level factories `blackjax.mclmc(...)` and `blackjax.adjusted_mclmc_dynamic(...)` (single-arg `integration_steps_fn`) are unchanged from 1.5.
  - The default `integrator` of both `adjusted_mclmc_dynamic.build_kernel` and `adjusted_mclmc.build_kernel` IS `isokinetic_mclachlan` (verified by identity check) — passing it explicitly is behavior-preserving self-documentation.
  - `blackjax.nuts.build_kernel()`/`.init`, `blackjax.adaptive_tempered_smc`, `blackjax.smc.extend_params`, `smc.init/step`: signatures unchanged — no code changes at those call sites.

## File Structure

- `src/gensbi/inference/samplers.py` — the ONLY source file modified (three methods: `MCLMC._run_unadjusted`, `MCLMC._run_adjusted`, `TemperedSMC._inner`; Task 4 may additionally touch `TemperedSMC.__init__` defaults + class docstring).
- `tests/inference/test_mclmc.py`, `tests/inference/test_smc.py` — existing tests, NOT modified; they are the acceptance gates.
- Scratch diagnostics (Task 4) live in the session scratchpad, never committed.

---

### Task 1: Port `MCLMC._run_unadjusted` (root cause A — 5 tests)

**Files:**
- Modify: `src/gensbi/inference/samplers.py:190-210` (`_run_unadjusted` method)

**Interfaces:**
- Consumes: `PosteriorTarget.log_posterior`, `target.prior.sample`, module helpers `_inference_loop`, `MclmcInfo` (all unchanged).
- Produces: `_run_unadjusted(key, target) -> (positions, MclmcInfo)` — same contract as before, now 1.6-compatible. No other task depends on its internals.

- [ ] **Step 1: Run the failing tests (RED evidence)**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v -k "unadjusted or return_info"`
Expected: 5 FAIL (`test_unadjusted_prior_recovery_real_flow`, `test_unadjusted_shape_and_finite`, `test_unadjusted_analytic_gaussian_recovery`, `test_unadjusted_multichain_shape`, `test_return_info`), each with
`ValueError: logdensity_fn is required. Pass the log-density function of the target distribution.` raised from `blackjax/adaptation/mclmc_adaptation.py`.

- [ ] **Step 2: Replace the method body**

Replace the whole `_run_unadjusted` method (currently `samplers.py:190-210`) with:

```python
    def _run_unadjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        pos_key, mom_key, tune_key, run_key = jax.random.split(key, 4)
        position = target.prior.sample(pos_key, ())
        init_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=target.log_posterior, rng_key=mom_key)
        # blackjax >= 1.6: build_kernel no longer binds logdensity_fn /
        # inverse_mass_matrix; the tuner threads them through per call.
        kernel = blackjax.mcmc.mclmc.build_kernel(integrator=isokinetic_mclachlan)
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, logdensity_fn=target.log_posterior,
            num_steps=self.num_tuning_steps, state=init_state,
            rng_key=tune_key, diagonal_preconditioning=self.diagonal_preconditioning)
        alg = blackjax.mclmc(target.log_posterior, L=params.L, step_size=params.step_size,
                             inverse_mass_matrix=params.inverse_mass_matrix)
        states, _ = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=float(params.L), step_size=float(params.step_size),
                         acceptance_rate=float(jnp.nan), num_samples=self.num_samples,
                         num_chains=self.num_chains)
        return states.position, info
```

(The only changes vs the old body: `build_kernel(integrator=...)` built once — the `lambda inverse_mass_matrix:` closure is deleted — and `logdensity_fn=target.log_posterior` added to `mclmc_find_L_and_step_size`. Everything else byte-identical.)

- [ ] **Step 3: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: **12 passed, 2 failed** — the 5 unadjusted tests now PASS alongside the 7 already-passing ones (`test_rescale_gives_exact_mean_step_count` ×4 params, `test_rescale_matches_blackjax_reference_at_15`, `test_check_rescale_domain_guard`, `test_adjusted_is_the_default`); `test_adjusted_analytic_gaussian_recovery` and `test_adjusted_reports_acceptance_rate` still FAIL with `TypeError: adjusted_mclmc_find_L_and_step_size() missing 1 required positional argument: 'logdensity_fn'` (that is Task 2's target — do not touch it here).

- [ ] **Step 4: Commit**

```bash
git add src/gensbi/inference/samplers.py
git commit -m "fix(inference): port unadjusted MCLMC to blackjax 1.6 API"
```

---

### Task 2: Port `MCLMC._run_adjusted` (root cause B — 2 tests)

**Files:**
- Modify: `src/gensbi/inference/samplers.py:212-245` (`_run_adjusted` method)

**Interfaces:**
- Consumes: module helpers `_rescale(mu)`, `_check_rescale_domain(mu)`, `_inference_loop`, `MclmcInfo` (all unchanged).
- Produces: `_run_adjusted(key, target) -> (positions, MclmcInfo)` — same contract, 1.6-compatible. No other task depends on its internals.

- [ ] **Step 1: Run the failing tests (RED evidence)**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v -k "test_adjusted and not default"`
(Note: plain `-k "adjusted"` would also match the `test_unadjusted_*` names by substring — keep the `test_adjusted` prefix.)
Expected: 2 FAIL (`test_adjusted_analytic_gaussian_recovery`, `test_adjusted_reports_acceptance_rate`) with `TypeError: adjusted_mclmc_find_L_and_step_size() missing 1 required positional argument: 'logdensity_fn'` at `samplers.py:229`.

- [ ] **Step 2: Replace the method body**

Replace the whole `_run_adjusted` method (currently `samplers.py:212-245`) with:

```python
    def _run_adjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        pos_key, init_key, tune_key, run_key = jax.random.split(key, 4)
        position = target.prior.sample(pos_key, ())
        init_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=position, logdensity_fn=target.log_posterior,
            random_generator_arg=init_key)

        # blackjax >= 1.6: the tuner drives the kernel as
        # kernel(rng_key=..., state=..., logdensity_fn=..., step_size=...,
        # inverse_mass_matrix=..., integration_steps_params=(avg,)); the
        # running average number of integration steps arrives as
        # integration_steps_fn's second positional argument.
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
        _check_rescale_domain(params.L / params.step_size)

        alg = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn=target.log_posterior, step_size=params.step_size,
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * _rescale(params.L / params.step_size)),
            inverse_mass_matrix=params.inverse_mass_matrix)

        states, infos = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=float(params.L), step_size=float(params.step_size),
                         acceptance_rate=float(jnp.mean(infos.acceptance_rate)),
                         num_samples=self.num_samples, num_chains=self.num_chains)
        return states.position, info
```

(Changes vs the old body: the per-call `def kernel(...)` closure that re-built `build_kernel` on every tuner step is replaced by one `build_kernel` call with a two-argument `integration_steps_fn`; `logdensity_fn=` added to the tuner call; explicit `integrator=isokinetic_mclachlan` — identical to the 1.6 default, kept for self-documentation. The post-tuning factory call, `_check_rescale_domain`, and info assembly are byte-identical to before.)

- [ ] **Step 3: Run the full MCLMC test file**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: **14/14 PASS**. Notes:
- Under the rewritten 1.6 tuner, `params.L / params.step_size` anchors at ≈ 2.0 (verified by smoke run), so `_check_rescale_domain` passes and `_rescale` gets mu ≈ 2 instead of 1.5-era larger values. The analytic recovery test (mean/var of an exact MH chain) and the acceptance-rate bounds test are expected to pass unchanged.
- If `test_adjusted_analytic_gaussian_recovery` fails numerically after a faithful port, that signals a porting bug (e.g. `integration_steps_fn` arity mismatch silently mis-stepping) — STOP and report with the numbers; do NOT loosen tolerances.

- [ ] **Step 4: Check docstrings for stale 1.5 tuner references**

Read the docstrings of `MCLMC` (class), `_rescale`, and `_check_rescale_domain` in `samplers.py`. The `_rescale` formula and its blackjax reference are version-independent (the formula did not change in 1.6) — expected outcome: no edits needed. If a docstring asserts 1.5-specific tuner *behavior* (e.g. claims about typical tuned `L/step_size` magnitudes), update that sentence to say the 1.6 tuner anchors `L/step_size` near 2.0; otherwise leave everything untouched.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py
git commit -m "fix(inference): port adjusted MCLMC to blackjax 1.6 API"
```

---

### Task 3: Port `TemperedSMC._inner` mclmc branch (root cause C — unblocks 2 tests)

**Files:**
- Modify: `src/gensbi/inference/samplers.py:318-335` (the `if self.inner_kernel == "mclmc":` branch of `_inner`)

**Interfaces:**
- Consumes: nothing new; `_inner` is called only by `TemperedSMC.run` (unchanged).
- Produces: `step_fn(rng_key, state, logdensity_fn, step_size, num_integration_steps, inverse_mass_matrix)` with the SAME signature and `params` dict keys as before — `TemperedSMC.run`'s `extend_params(params)` threading depends on those key names (`step_size`, `num_integration_steps`, `inverse_mass_matrix`). Do not rename them.

- [ ] **Step 1: Run the failing tests (RED evidence)**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v -k "mclmc"`
Expected: `test_smc_mclmc_is_the_default_inner_kernel` PASS; `test_smc_mclmc_recovers_both_modes` and `test_smc_mclmc_analytic_gaussian_recovery` FAIL with `TypeError: build_kernel() got an unexpected keyword argument 'logdensity_fn'` at `samplers.py:325`.

- [ ] **Step 2: Replace the mclmc branch**

In `_inner`, replace the `if self.inner_kernel == "mclmc":` block (currently `samplers.py:318-335`) with:

```python
        if self.inner_kernel == "mclmc":
            from blackjax.mcmc.integrators import isokinetic_mclachlan

            # blackjax >= 1.6: build_kernel is a pure factory (no logdensity_fn /
            # inverse_mass_matrix binding), so build it once; SMC injects the
            # tempered logdensity per call.
            kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                integrator=isokinetic_mclachlan)

            def step_fn(rng_key, state, logdensity_fn, step_size,
                        num_integration_steps, inverse_mass_matrix):
                return kernel(rng_key, state, logdensity_fn, step_size,
                              integration_steps_params=(num_integration_steps,),
                              inverse_mass_matrix=inverse_mass_matrix)

            init_fn = blackjax.mcmc.adjusted_mclmc.init   # (position, logdensity_fn) -> HMCState
            params = dict(step_size=self.inner_step_size,
                          num_integration_steps=self.inner_num_integration_steps,
                          inverse_mass_matrix=imm)
            return step_fn, init_fn, params
```

(Changes vs the old block: `build_kernel(integrator=...)` hoisted outside `step_fn` — no more per-SMC-step rebuild — and the kernel call passes `logdensity_fn` positionally plus `integration_steps_params=(num_integration_steps,)` instead of the removed `num_integration_steps=` kwarg. `step_fn`'s own signature and the `params` dict are byte-identical to before.)

- [ ] **Step 3: Run the SMC test file and RECORD outcomes**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v`
Expected with certainty: no `TypeError` anywhere; `test_smc_mclmc_is_the_default_inner_kernel` and `test_smc_unknown_kernel_raises` PASS; `test_smc_nuts_analytic_gaussian_recovery` still FAILS on its mean-recovery assert (root cause D — Task 4's target, pre-existing, NOT caused by this task).
Uncertain (record, don't fix): the three other recovery tests (`test_smc_mclmc_recovers_both_modes`, `test_smc_mclmc_analytic_gaussian_recovery`, `test_smc_nuts_recovers_both_modes`) and `test_smc_info_has_log_evidence` use the same 1.6 adaptive-tempering machinery whose numerics changed upstream — they may pass or fail numerically. Record each test's pass/fail and, for failures, the observed vs expected numbers in your report. Numeric failures here are Task 4's scope; do NOT retune or edit anything in this task.

- [ ] **Step 4: Commit**

Gate for committing: zero `TypeError`s and the two certainty tests green (Step 3). Known-numeric failures are tracked plan scope (Task 4) and do not block this commit.

```bash
git add src/gensbi/inference/samplers.py
git commit -m "fix(inference): port TemperedSMC inner MCLMC kernel to blackjax 1.6 API"
```

---

### Task 4: Diagnose + retune adaptive tempering (root cause D — remaining SMC failures)

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (ONLY `TemperedSMC.__init__` default values and the class docstring, IF diagnosis says defaults must change; no other code)
- Scratch (not committed): diagnostic scripts in the session scratchpad

**Interfaces:**
- Consumes: Task 3's ported `_inner`; `SmcInfo.num_temperature_steps` (already exposed — the key diagnostic).
- Produces: all 7 tests in `tests/inference/test_smc.py` passing with genuine posterior recovery; updated `TemperedSMC` defaults + docstrings if retuned.

Context: blackjax 1.6 fixed a sign bug in the adaptive-tempering ESS bisection target (`smc/ess.py`, upstream #914: `-delta * logdensity` → `delta * logdensity`). Under 1.6, `test_smc_nuts_analytic_gaussian_recovery` (same code + seed that passes under 1.5) returns posterior mean ≈ (0.03, 0.02) instead of (0.5, −0.5) — prior-like samples. Working hypothesis: the corrected solver anneals λ 0→1 in very few temperature steps, so `num_mcmc_steps=10` of rejuvenation cannot move particles to the posterior. The acceptance bar is REAL recovery, not loosened tolerances.

- [ ] **Step 1: Diagnose — λ schedule under 1.5 vs 1.6**

Write the scratchpad script `smc_diag.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import NLEPosterior, TemperedSMC


class GaussianMock:
    def log_prob(self, x, cond):
        diff = (x - cond).reshape(x.shape[0], -1)
        return -0.5 * jnp.sum(diff ** 2, axis=-1)


dim = 2
x_o = jnp.array([1.0, -1.0])
post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
s, info = post.sample(jax.random.PRNGKey(0), x_o,
                      sampler=TemperedSMC(inner_kernel="nuts", num_particles=2000,
                                          inner_step_size=0.5),
                      return_info=True)
m = jnp.mean(s[..., 0], axis=0)
print(f"n_temp_steps={info.num_temperature_steps}  logZ={info.log_evidence:.3f}  "
      f"final_beta={info.final_tempering_param}  mean={m}  target={x_o / 2}")
```

Run it in BOTH installs (same script, same seed):
- 1.6: `JAX_PLATFORMS=cpu mamba run -n gensbi python <scratchpad>/smc_diag.py`
- 1.5 (comparison only): `JAX_PLATFORMS=cpu .venv/bin/python <scratchpad>/smc_diag.py` from the repo root

Expected: the 1.5 run recovers mean ≈ (0.5, −0.5); compare `n_temp_steps` between the two. If 1.6's `n_temp_steps` is much smaller (e.g. 1–3 vs 1.5's many), the anneals-too-fast hypothesis is CONFIRMED. If `n_temp_steps` is similar but recovery still fails, the hypothesis is wrong — proceed to Step 2's audit before touching any knob, and report what you find.

- [ ] **Step 2: Audit blackjax 1.6 SMC-side default changes**

Diff the SMC machinery between the two installed trees (read-only):

```bash
V15=.venv/lib/python3.12/site-packages/blackjax
V16=/lhome/ific/a/aamerio/miniforge3/envs/gensbi/lib/python3.12/site-packages/blackjax
diff -u $V15/smc/adaptive_tempered.py $V16/smc/adaptive_tempered.py
diff -u $V15/smc/ess.py $V16/smc/ess.py
diff -u $V15/smc/solver.py $V16/smc/solver.py
```

Record in your report: any changed defaults (solver iteration caps, `target_ess` conventions, resampling thresholds) beyond the known #914 sign fix, and whether any of them implies a specific retune (e.g. if `target_ess` semantics inverted, that IS the fix — adjust GenSBI's passthrough interpretation accordingly and say so).

- [ ] **Step 3: Retune grid on the failing NUTS analytic case**

Extend `smc_diag.py` with a small grid over GenSBI's own knobs (run under 1.6 only):

```python
for target_ess in (0.5, 0.7, 0.9):
    for num_mcmc_steps in (10, 30, 50):
        s, info = post.sample(jax.random.PRNGKey(0), x_o,
                              sampler=TemperedSMC(inner_kernel="nuts", num_particles=2000,
                                                  inner_step_size=0.5,
                                                  target_ess=target_ess,
                                                  num_mcmc_steps=num_mcmc_steps),
                              return_info=True)
        m = jnp.mean(s[..., 0], axis=0)
        err = float(jnp.max(jnp.abs(m - x_o / 2)))
        print(f"target_ess={target_ess} num_mcmc_steps={num_mcmc_steps}: "
              f"steps={info.num_temperature_steps} err={err:.3f}")
```

Decision rule: pick the SMALLEST deviation from the current defaults (`target_ess=0.5`, `num_mcmc_steps=10`) whose `err` ≤ 0.1 (comfortable margin inside the test's 0.2 tolerance). Preference order: bump `num_mcmc_steps` first (pure rejuvenation cost, no schedule change), raise `target_ess` second (finer λ ladder). Then re-run the chosen combination with a DIFFERENT seed (`PRNGKey(42)`) to confirm it isn't seed luck (require `err` ≤ 0.15 there too).

- [ ] **Step 4: Apply the retuned defaults**

In `TemperedSMC.__init__` (currently `samplers.py:296-305`), change ONLY the default value(s) selected in Step 3, and update the matching `Parameters` lines in the class docstring (currently `samplers.py:277-293`) — e.g. if `num_mcmc_steps` default becomes 30, both the signature and the docstring's "Default is 10." line change. Add one sentence to the class docstring noting the defaults are calibrated for blackjax ≥ 1.6's corrected adaptive tempering (upstream #914).

If Step 3 finds NO grid point that recovers (all `err` > 0.1): STOP and report BLOCKED with the full grid table — do not edit test files, do not loosen tolerances, do not invent new knobs.

- [ ] **Step 5: Run the full SMC test file**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v`
Expected: **7/7 PASS** — including any tests Task 3 recorded as numerically failing (the retuned defaults govern them too; they pass explicit `inner_step_size`/`inner_num_integration_steps` but inherit `target_ess`/`num_mcmc_steps` defaults). If an mclmc-inner recovery test still fails, extend the Step 3 grid to that exact failing configuration (same knobs the test passes, varying only the retune levers) and iterate once; if it still fails, STOP and report BLOCKED with numbers.

- [ ] **Step 6: Run the complete inference suite (plan gate)**

Run: `JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest tests/inference/ -v`
Expected: **36 passed, 0 failed** (14 test_mclmc.py + 7 test_smc.py + 11 test_nested.py + 4 test_target.py).

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/inference/samplers.py
git commit -m "fix(inference): retune TemperedSMC defaults for blackjax 1.6 adaptive tempering"
```

(If Steps 1–3 concluded no default change is needed — e.g. the audit found a passthrough fix instead — adapt the commit message to what actually changed; there must still be a commit closing root cause D, and the report must document the diagnosis evidence either way.)
