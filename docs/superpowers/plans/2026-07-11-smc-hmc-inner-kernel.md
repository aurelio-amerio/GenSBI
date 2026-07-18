# SMC Inner Kernel: Swap NUTS → HMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `"nuts"` inner rejuvenation kernel option in `TemperedSMC` with `"hmc"`, since a fixed-trajectory kernel is the correct computational shape for SMC particle rejuvenation (static per-particle cost, vectorizes cleanly across particles) whereas NUTS's dynamic trajectory length is a liability under particle batching.

**Architecture:** `TemperedSMC._inner()` is a small factory returning `(mcmc_step_fn, mcmc_init_fn, mcmc_parameters)` for blackjax's `adaptive_tempered_smc`. The change is confined to that method plus two docstrings. HMC is a drop-in for NUTS: `blackjax.hmc.build_kernel()` returns a kernel whose signature is `(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)`, so the SMC framework can call it directly with `mcmc_parameters` extended by the extra `num_integration_steps` key — no wrapper closure needed (unlike the MCLMC branch). `TemperedSMC` already carries `inner_num_integration_steps` (default 5), so no new constructor parameter is required.

**Tech Stack:** Python 3.12, JAX, blackjax 1.6, pytest.

## Global Constraints

- blackjax version floor: **1.6** (kernel factories are pure; `build_kernel()` takes no `logdensity_fn`/`inverse_mass_matrix` binding).
- Tests run CPU-only: the test module sets `os.environ["JAX_PLATFORMS"] = "cpu"` at import (top of `tests/inference/test_smc.py`). On this GPU box the GPUs are frequently busy, so any manual repro must prefix `JAX_PLATFORMS=cpu` and use the `gensbi` mamba/conda env (not `.venv`).
- Valid `inner_kernel` values after this change: exactly `"mclmc"` (default) and `"hmc"`. `"nuts"` must raise `ValueError` via the existing unknown-kernel branch.
- This is un-merged experimental work on branch `nested-sampling`; no deprecation shim for `"nuts"` is required — a hard `ValueError` is acceptable (no external users).

---

### Task 1: Swap the NUTS inner kernel for HMC

**Files:**
- Modify: `src/gensbi/inference/samplers.py` — `TemperedSMC` class docstring (approx. lines 277–309), `_inner` method (approx. lines 322–353), `inner_num_integration_steps` param docstring (approx. line 304–305).
- Test: `tests/inference/test_smc.py` — migrate the three `inner_kernel="nuts"` tests (approx. lines 34–66) to `"hmc"`.

**Interfaces:**
- Consumes: `TemperedSMC.inner_step_size` (float, default 0.1), `TemperedSMC.inner_num_integration_steps` (int, default 5), `TemperedSMC.inner_inverse_mass_matrix` (Array | None), `target.dim` (int) — all already present, unchanged.
- Produces: `TemperedSMC(inner_kernel="hmc")` yields a working sampler; `_inner` returns `(step_fn, init_fn, params)` where for HMC `params = dict(step_size, num_integration_steps, inverse_mass_matrix)`. `inner_kernel="nuts"` now raises `ValueError`. Public constructor signature is unchanged.

- [ ] **Step 1: Migrate the three NUTS tests to HMC (write the failing tests)**

In `tests/inference/test_smc.py`, replace the three NUTS-based tests. Rename the two recovery tests, switch `inner_kernel="nuts"` → `inner_kernel="hmc"`, and give the recovery tests `inner_num_integration_steps=10` (mirroring the MCLMC recovery tests, so a fixed 10-step trajectory does enough work per temperature). The bimodal-recovery assertion stays the same — SMC's population, not the inner kernel, is what captures both modes.

Replace lines 34–66 (the `test_smc_nuts_analytic_gaussian_recovery`, `test_smc_nuts_recovers_both_modes`, and `test_smc_info_has_log_evidence` functions) with:

```python
def test_smc_hmc_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(0), x_o,
                    sampler=TemperedSMC(inner_kernel="hmc", num_particles=2000,
                                        inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    assert s.shape == (2000, dim)
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.2)


def test_smc_hmc_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=TemperedSMC(inner_kernel="hmc", num_particles=2000,
                                        inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # both modes populated (a single MCMC chain would capture only one)
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_smc_info_has_log_evidence():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    _, info = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                          sampler=TemperedSMC(inner_kernel="hmc", num_particles=1000,
                                              inner_step_size=0.5,
                                              inner_num_integration_steps=10),
                          return_info=True)
    assert jnp.isfinite(info.log_evidence)
    assert info.num_temperature_steps > 0
    assert jnp.isclose(info.final_tempering_param, 1.0, atol=1e-6)
```

Leave `test_smc_unknown_kernel_raises` (still uses `inner_kernel="foo"`), `test_smc_mclmc_is_the_default_inner_kernel`, and the two `mclmc` tests unchanged.

- [ ] **Step 2: Run the HMC tests to verify they fail**

Run:
```bash
JAX_PLATFORMS=cpu python -m pytest tests/inference/test_smc.py::test_smc_hmc_analytic_gaussian_recovery tests/inference/test_smc.py::test_smc_hmc_recovers_both_modes tests/inference/test_smc.py::test_smc_info_has_log_evidence -v
```
Expected: FAIL — `_inner` has no `"hmc"` branch, so it falls through to `raise ValueError(f"unknown inner_kernel {self.inner_kernel!r}")`. Each test should error with a `ValueError` mentioning `'hmc'`, not an assertion failure. (If instead they pass, the `"hmc"` branch already exists — stop and reconcile.)

- [ ] **Step 3: Add the HMC branch to `_inner` and remove the NUTS branch**

In `src/gensbi/inference/samplers.py`, in `TemperedSMC._inner`, replace the NUTS branch:

```python
        if self.inner_kernel == "nuts":
            step_fn = blackjax.nuts.build_kernel()
            init_fn = blackjax.nuts.init
            params = dict(step_size=self.inner_step_size, inverse_mass_matrix=imm)
            return step_fn, init_fn, params
```

with the HMC branch:

```python
        if self.inner_kernel == "hmc":
            # blackjax >= 1.6: build_kernel() is a pure factory; the returned
            # kernel's signature is (rng_key, state, logdensity_fn, step_size,
            # inverse_mass_matrix, num_integration_steps), so SMC can call it
            # directly with these params as kwargs -- no wrapper needed. HMC's
            # fixed num_integration_steps gives static per-particle cost that
            # vectorizes cleanly across the particle population (unlike NUTS's
            # data-dependent trajectory length).
            step_fn = blackjax.hmc.build_kernel()
            init_fn = blackjax.hmc.init
            params = dict(step_size=self.inner_step_size,
                          num_integration_steps=self.inner_num_integration_steps,
                          inverse_mass_matrix=imm)
            return step_fn, init_fn, params
```

Leave the `"mclmc"` branch and the trailing `raise ValueError(f"unknown inner_kernel {self.inner_kernel!r}")` untouched.

- [ ] **Step 4: Update the class and parameter docstrings**

In the `TemperedSMC` class docstring, change the kernel-summary sentence:

```
    The inner rejuvenation kernel is adjusted MCLMC by default; NUTS is available
    as a fallback.
```
to:
```
    The inner rejuvenation kernel is adjusted MCLMC by default; fixed-trajectory
    HMC is available as an alternative. (NUTS is deliberately not offered: its
    data-dependent trajectory length does not vectorize cleanly across SMC
    particles, and rejuvenation does not need NUTS's full-mixing guarantee.)
```

Change the `inner_kernel` parameter doc:
```
    inner_kernel : str, optional
        Inner MCMC kernel: ``"mclmc"`` (adjusted MCLMC, default) or
        ``"nuts"``.
```
to:
```
    inner_kernel : str, optional
        Inner MCMC kernel: ``"mclmc"`` (adjusted MCLMC, default) or
        ``"hmc"`` (fixed-trajectory HMC).
```

Change the `inner_num_integration_steps` parameter doc:
```
    inner_num_integration_steps : int, optional
        Number of integration steps for the inner MCLMC kernel.  Default is 5.
```
to:
```
    inner_num_integration_steps : int, optional
        Number of integration steps for the inner MCLMC or HMC kernel.
        Default is 5.
```

- [ ] **Step 5: Run the full SMC test module to verify green**

Run:
```bash
JAX_PLATFORMS=cpu python -m pytest tests/inference/test_smc.py -v
```
Expected: PASS — all tests green, including the three migrated `hmc` tests, `test_smc_unknown_kernel_raises` (`"foo"` still raises), and the untouched `mclmc` tests. Output pristine (no errors/warnings beyond routine JAX platform notices).

- [ ] **Step 6: Grep to confirm no stray `nuts` references remain in the sampler surface**

Run:
```bash
grep -rn "nuts" -i src/gensbi/inference/samplers.py tests/inference/test_smc.py
```
Expected: no matches. (The unrelated diffusion/flow-matching notebooks that mention "nuts" are out of scope and untouched.)

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/inference/samplers.py tests/inference/test_smc.py
git commit -m "feat(inference): swap TemperedSMC inner NUTS kernel for fixed-trajectory HMC

NUTS's data-dependent trajectory length does not vectorize cleanly across
SMC particles and rejuvenation does not need its full-mixing guarantee; a
fixed-trajectory kernel (static per-particle cost) is the correct shape for
SMC. HMC drops in via blackjax.hmc.build_kernel() reusing the existing
inner_num_integration_steps. inner_kernel now accepts 'mclmc' (default) or
'hmc'; 'nuts' raises ValueError.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Remove NUTS → Step 3 deletes the branch; Step 6 verifies no residue. ✓
- Add HMC → Step 3 adds the branch using the verified blackjax 1.6 signature; Steps 1–2 cover it with failing-first tests. ✓
- `"nuts"` now errors → covered by the unchanged trailing `raise ValueError` and `test_smc_unknown_kernel_raises` (Step 5). ✓
- Docstrings reflect the new surface → Step 4. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" — every code step shows the exact code. ✓

**Type consistency:** HMC `params` keys (`step_size`, `num_integration_steps`, `inverse_mass_matrix`) match `blackjax.hmc.build_kernel()`'s kwargs verified via `inspect.signature`. `inner_num_integration_steps` and `inner_step_size` are existing `TemperedSMC` attributes. Test function names are unique and don't collide with retained tests. ✓
