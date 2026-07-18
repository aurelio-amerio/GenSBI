# Nested Sampling for NLE Posterior Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `NestedSampler` (blackjax ≥ 1.6 nested slice sampling) as an opt-in `Sampler` for `NLEPosterior`, returning equal-weight posterior samples plus log-evidence diagnostics.

**Architecture:** One new `Sampler` subclass + one frozen info dataclass in `src/gensbi/inference/samplers.py`, exported from `gensbi.inference`. `run(key, target)` follows the blackjax-canonical pattern: init live points from the prior, jitted `step` in a Python `while` loop until `logZ_live − logZ < dlogz`, then `finalise` → evidence/ESS → resample equal-weight draws. `posterior.py` is untouched — `PosteriorTarget.log_prior`/`log_likelihood` map 1:1 onto the `blackjax.nss` API.

**Tech Stack:** JAX, blackjax ≥ 1.6 (`blackjax.nss`, `blackjax.ns.utils`), pytest. Spec: `docs/superpowers/specs/2026-07-11-nested-sampling-design.md`.

## Global Constraints

- Tests run on CPU: every test file starts with `import os; os.environ["JAX_PLATFORMS"] = "cpu"` **before** importing jax (pattern from `tests/inference/test_smc.py`).
- Test command runs inside the mamba `gensbi` env (NOT `.venv`): `python -m pytest tests/inference/test_nested.py -v`.
- blackjax is imported **lazily inside `run()`** (module convention in `samplers.py`; keeps top-level import light).
- NumPy-style docstrings, matching the existing classes in `samplers.py`.
- API facts (verified against installed blackjax 1.6 by smoke run):
  - `blackjax.nss(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete)` → `SamplingAlgorithm`; both fns take a single particle `(dim,)`.
  - `algo.init(particles)` accepts `(num_live, dim)` positions directly.
  - Termination reads `state.integrator.logZ_live - state.integrator.logZ`.
  - `finalise(state, dead)` → `NSInfo`; total points = `ns_run.particles.loglikelihood.shape[0]` (dead + final live).
  - `log_weights(key, ns_run, shape=100)` → `(num_points, 100)`; logZ draws = `logsumexp` over axis 0.
  - `blackjax.ns.utils.sample(key, ns_run, n)` → resampled `StateWithLogLikelihood`; positions at `.position`, shape `(n, dim)`.

## File Structure

- `src/gensbi/inference/samplers.py` — append `NestedSamplerInfo` + `NestedSampler` after `TemperedSMC` (all samplers live in this one file by convention).
- `src/gensbi/inference/__init__.py` — export both names.
- `tests/inference/test_nested.py` — new test file; mocks copied from `test_smc.py` (that file keeps its mocks private, so we copy rather than import).

---

### Task 1: `NestedSamplerInfo` dataclass + `NestedSampler` constructor, validation, exports

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (append at end of file)
- Modify: `src/gensbi/inference/__init__.py`
- Test: `tests/inference/test_nested.py` (create)

**Interfaces:**
- Consumes: `Sampler` ABC (already in `samplers.py`).
- Produces: `NestedSampler(*, num_live=500, num_delete=None, num_inner_steps=None, num_samples=1000, dlogz=-3.0, max_iterations=100_000)` with attributes of the same names (`num_delete` resolved to `max(1, num_live // 10)` when `None`), method `_resolve_num_inner_steps(dim) -> int`, and frozen dataclass `NestedSamplerInfo(log_evidence, log_evidence_err, ess, num_dead, dead)`. Both importable from `gensbi.inference`.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_nested.py`:

```python
# tests/inference/test_nested.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import NLEPosterior, NestedSampler, NestedSamplerInfo


class GaussianMock:
    def log_prob(self, x, cond):
        diff = (x - cond).reshape(x.shape[0], -1)     # flatten all non-batch dims
        return -0.5 * jnp.sum(diff ** 2, axis=-1)     # (B,)


class BimodalMock:
    """log q(x | theta) = mixture of N(theta; +mu, 0.5 I) and N(theta; -mu, 0.5 I).

    Independent of the observation x; posterior under a broad prior is bimodal at +/-mu.
    """
    def __init__(self, mu=3.0, sigma=0.5):
        self.mu, self.sigma = mu, sigma

    def log_prob(self, x, cond):
        cf = cond.reshape(cond.shape[0], -1)           # flatten all non-batch dims
        a = -0.5 * jnp.sum(((cf - self.mu) / self.sigma) ** 2, axis=-1)
        b = -0.5 * jnp.sum(((cf + self.mu) / self.sigma) ** 2, axis=-1)
        return jax.scipy.special.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)


def test_constructor_defaults():
    s = NestedSampler()
    assert s.num_live == 500
    assert s.num_delete == 50                    # num_live // 10
    assert s.num_inner_steps is None             # resolved per-target at run time
    assert s.num_samples == 1000
    assert s.dlogz == -3.0
    assert s.max_iterations == 100_000


def test_num_inner_steps_auto_resolution():
    s = NestedSampler()
    assert s._resolve_num_inner_steps(2) == 5    # max(5, 2*2)
    assert s._resolve_num_inner_steps(10) == 20  # max(5, 2*10)
    assert NestedSampler(num_inner_steps=7)._resolve_num_inner_steps(10) == 7


def test_num_delete_floor_is_one():
    assert NestedSampler(num_live=5).num_delete == 1   # max(1, 5 // 10)


def test_constructor_validation():
    with pytest.raises(ValueError):
        NestedSampler(num_live=0)
    with pytest.raises(ValueError):
        NestedSampler(num_live=10, num_delete=10)      # must be < num_live
    with pytest.raises(ValueError):
        NestedSampler(num_live=10, num_delete=0)


def test_info_dataclass_is_frozen():
    info = NestedSamplerInfo(log_evidence=0.0, log_evidence_err=0.1,
                             ess=100.0, num_dead=500, dead=None)
    with pytest.raises(Exception):   # dataclasses.FrozenInstanceError
        info.log_evidence = 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_nested.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'NestedSampler' from 'gensbi.inference'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gensbi/inference/samplers.py` (after `TemperedSMC`):

```python
@dataclass(frozen=True)
class NestedSamplerInfo:
    """Diagnostics from a nested sampling run.

    Parameters
    ----------
    log_evidence : float
        Log marginal likelihood estimate (mean over stochastic
        prior-volume draws).
    log_evidence_err : float
        Standard deviation of the log-evidence estimate over the
        stochastic prior-volume draws.
    ess : float
        Effective sample size of the weighted dead-point set.
    num_dead : int
        Total number of points in the finalised run (dead points plus
        the final live set).
    dead : object
        Raw finalised ``blackjax.ns.base.NSInfo`` carrying the full
        point history (positions, log-likelihoods, birth contours).
        Kept for downstream re-weighting or anesthetic-style analysis.
    """

    log_evidence: float
    log_evidence_err: float
    ess: float
    num_dead: int
    dead: object


class NestedSampler(Sampler):
    """Nested slice sampling (blackjax ``nss``) posterior sampler.

    Runs blackjax's Nested Slice Sampling from prior-drawn live points,
    accumulating dead points until the live set's evidence share is
    negligible, then resamples the dead-point history into equal-weight
    posterior draws.  Unlike the MCMC samplers this also estimates the
    log evidence, and handles multimodal posteriors without tempering.

    Parameters
    ----------
    num_live : int, optional
        Number of live points.  Default is 500.
    num_delete : int or None, optional
        Number of lowest-likelihood points replaced per step (device
        batching).  If ``None``, defaults to ``max(1, num_live // 10)``.
    num_inner_steps : int or None, optional
        Constrained slice moves per replacement.  If ``None``, resolved
        at run time to ``max(5, 2 * target.dim)`` (blackjax's rule of
        thumb for reliable mixing).  Default is ``None``.
    num_samples : int, optional
        Number of equal-weight posterior draws returned.  Default is 1000.
    dlogz : float, optional
        Termination threshold (blackjax convention): stop once
        ``logZ_live - logZ < dlogz``.  Default is -3.0; use e.g. -10.0
        near phase transitions.
    max_iterations : int, optional
        Safety cap on the number of outer NS steps.  Default is 100_000.
    """

    def __init__(self, *, num_live=500, num_delete=None, num_inner_steps=None,
                 num_samples=1000, dlogz=-3.0, max_iterations=100_000):
        if num_live <= 0:
            raise ValueError(f"num_live must be positive, got {num_live}")
        if num_delete is None:
            num_delete = max(1, num_live // 10)
        if not 1 <= num_delete < num_live:
            raise ValueError(
                f"num_delete must be in [1, num_live), got num_delete="
                f"{num_delete} with num_live={num_live}")
        self.num_live = num_live
        self.num_delete = num_delete
        self.num_inner_steps = num_inner_steps
        self.num_samples = num_samples
        self.dlogz = dlogz
        self.max_iterations = max_iterations

    def _resolve_num_inner_steps(self, dim):
        """``num_inner_steps`` if set, else blackjax's ``max(5, 2 * dim)``."""
        if self.num_inner_steps is not None:
            return self.num_inner_steps
        return max(5, 2 * dim)

    def run(self, key, target):
        raise NotImplementedError("implemented in Task 2")
```

(`Sampler.run` is an `@abstractmethod`, so `NestedSampler` needs this
temporary stub to be instantiable for Task 1's tests; Task 2 replaces it.)

Update `src/gensbi/inference/__init__.py`:

```python
"""Inference wrappers: NLE posterior sampling over trained density flows."""

from gensbi.inference.posterior import NLEPosterior, PosteriorTarget
from gensbi.inference.samplers import (Sampler, MCLMC, MclmcInfo, TemperedSMC,
                                       SmcInfo, NestedSampler, NestedSamplerInfo)

__all__ = ["NLEPosterior", "PosteriorTarget", "Sampler",
           "MCLMC", "MclmcInfo", "TemperedSMC", "SmcInfo",
           "NestedSampler", "NestedSamplerInfo"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/inference/test_nested.py -v`
Expected: 5 PASS (`test_constructor_defaults`, `test_num_inner_steps_auto_resolution`, `test_num_delete_floor_is_one`, `test_constructor_validation`, `test_info_dataclass_is_frozen`)

Also run the existing suite to catch export regressions:
Run: `python -m pytest tests/inference/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py src/gensbi/inference/__init__.py tests/inference/test_nested.py
git commit -m "feat(inference): NestedSampler scaffold — constructor, validation, info dataclass"
```

---

### Task 2: `NestedSampler.run()` — NS loop, evidence, resampling

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (add `run` to `NestedSampler` from Task 1)
- Test: `tests/inference/test_nested.py` (append)

**Interfaces:**
- Consumes: `PosteriorTarget` fields `log_prior(theta)->float`, `log_likelihood(theta)->float`, `prior.sample(key, (n,))->(n, dim)`, `dim: int`; Task 1's `NestedSamplerInfo` and `_resolve_num_inner_steps(dim)`.
- Produces: `run(key, target) -> (samples, info)` with `samples` shape `(num_samples, dim)` and `info: NestedSamplerInfo`. This satisfies the `Sampler` contract consumed by `NLEPosterior.sample`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/inference/test_nested.py`. A module-scoped fixture runs NS once and shares the result across the three assertions (an NS run takes seconds on CPU; don't repeat it per test):

```python
@pytest.fixture(scope="module")
def gaussian_ns_run():
    """One shared NS run on the analytic 2D Gaussian target."""
    dim = 2
    x_o = jnp.array([1.0, -1.0])
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    target = post.build_target(x_o)
    samples, info = NestedSampler().run(jax.random.PRNGKey(0), target)
    return dim, x_o, samples, info


def test_gaussian_recovery_and_shape(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    assert samples.shape == (1000, dim)          # num_samples default
    # prior N(0, I), likelihood N(x_o; theta, I) -> posterior mean x_o / 2
    assert jnp.allclose(jnp.mean(samples, axis=0), x_o / 2, atol=0.2)


def test_evidence_matches_analytic(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    # GaussianMock omits the Gaussian normalisation constant, so
    # log Z = -||x_o||^2 / 4 - (dim / 2) * log 2   (see spec, Testing #3)
    logZ_true = -jnp.sum(x_o ** 2) / 4 - dim / 2 * jnp.log(2.0)
    tol = max(3.0 * info.log_evidence_err, 0.3)  # floor guards tiny stochastic-volume err
    assert abs(info.log_evidence - logZ_true) < tol


def test_info_contract(gaussian_ns_run):
    dim, x_o, samples, info = gaussian_ns_run
    assert isinstance(info, NestedSamplerInfo)
    assert jnp.isfinite(info.log_evidence)
    assert info.log_evidence_err > 0
    assert info.ess > 0
    assert info.num_dead > 0
    assert info.dead is not None                 # raw finalised NSInfo retained
    assert jnp.all(jnp.isfinite(samples))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_nested.py -v`
Expected: the 3 new tests FAIL with `NotImplementedError: implemented in Task 2` (raised by Task 1's stub inside the `gaussian_ns_run` fixture); Task 1's 5 tests still PASS.

- [ ] **Step 3: Write the implementation**

Replace Task 1's stub `run` in `NestedSampler` with:

```python
    def run(self, key, target):
        """Draw posterior samples using nested slice sampling.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        target : PosteriorTarget
            Posterior target produced by
            :meth:`~gensbi.inference.posterior.NLEPosterior.build_target`.

        Returns
        -------
        samples : Array
            Equal-weight posterior samples of shape ``(num_samples, dim)``.
        info : NestedSamplerInfo
            Evidence estimate, ESS and the raw finalised run.
        """
        import blackjax
        from blackjax.ns.utils import ess, finalise, log_weights
        from blackjax.ns.utils import sample as ns_sample

        init_key, run_key, weights_key, ess_key, resample_key = \
            jax.random.split(key, 5)
        particles = target.prior.sample(init_key, (self.num_live,))
        algo = blackjax.nss(
            logprior_fn=target.log_prior,
            loglikelihood_fn=target.log_likelihood,
            num_inner_steps=self._resolve_num_inner_steps(target.dim),
            num_delete=self.num_delete)
        state = algo.init(particles)
        step = jax.jit(algo.step)

        # blackjax-canonical loop: dead points accumulate with variable
        # length, so this stays a Python loop over a jitted step.
        dead = []
        while state.integrator.logZ_live - state.integrator.logZ >= self.dlogz:
            run_key, subkey = jax.random.split(run_key)
            state, step_info = step(subkey, state)
            dead.append(step_info)

        ns_run = finalise(state, dead)
        logw = log_weights(weights_key, ns_run, shape=100)   # (num_points, 100)
        logz_draws = jax.scipy.special.logsumexp(logw, axis=0)
        samples = ns_sample(resample_key, ns_run, self.num_samples).position
        info = NestedSamplerInfo(
            log_evidence=float(jnp.mean(logz_draws)),
            log_evidence_err=float(jnp.std(logz_draws)),
            ess=float(ess(ess_key, ns_run)),
            num_dead=int(ns_run.particles.loglikelihood.shape[0]),
            dead=ns_run,
        )
        return samples, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/inference/test_nested.py -v`
Expected: 8 PASS. (Reference values from the pre-plan smoke run of the identical loop: logZ ≈ −1.23 ± 0.03 vs analytic −1.193; posterior mean ≈ (0.50, −0.55); ESS ≈ 1600.)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py tests/inference/test_nested.py
git commit -m "feat(inference): NestedSampler.run — blackjax nss loop, evidence, equal-weight resampling"
```

---

### Task 3: `max_iterations` guard, bimodal recovery, pipeline wiring

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (guard inside `NestedSampler.run`)
- Test: `tests/inference/test_nested.py` (append)

**Interfaces:**
- Consumes: Task 2's `run` loop; `NLEPosterior.sample(key, x_o, sampler=..., return_info=...)` (existing, `posterior.py:107`).
- Produces: `RuntimeError` on loop overrun; no new public API.

- [ ] **Step 1: Write the failing tests**

Append to `tests/inference/test_nested.py`:

```python
def test_max_iterations_guard_raises():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    target = post.build_target(jnp.array([1.0, -1.0]))
    with pytest.raises(RuntimeError, match="max_iterations"):
        NestedSampler(max_iterations=2).run(jax.random.PRNGKey(0), target)


def test_bimodal_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=NestedSampler(num_samples=2000))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # both modes populated (a single MCMC chain would capture only one)
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_pipeline_wiring_shapes_and_info():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    samples, info = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                                sampler=NestedSampler(num_samples=200),
                                return_info=True)
    assert samples.shape == (200, dim, 1)        # NLEPosterior expands (n, dim) -> (n, dim, 1)
    assert isinstance(info, NestedSamplerInfo)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_nested.py -v -k "max_iterations or bimodal or wiring"`
Expected: `test_max_iterations_guard_raises` FAILS (no `RuntimeError` raised — with `max_iterations=2` the loop simply keeps running until natural termination, so the test fails on `DID NOT RAISE`). The other two should PASS already (they exercise Task 2 code through the pipeline); if they fail, that is a real bug in Task 2 — stop and fix it before proceeding.

- [ ] **Step 3: Implement the guard**

In `NestedSampler.run`, replace the loop from Task 2 with:

```python
        dead = []
        while state.integrator.logZ_live - state.integrator.logZ >= self.dlogz:
            if len(dead) >= self.max_iterations:
                raise RuntimeError(
                    f"nested sampling did not terminate within max_iterations="
                    f"{self.max_iterations} steps (logZ_live - logZ = "
                    f"{float(state.integrator.logZ_live - state.integrator.logZ):.3g}"
                    f" >= dlogz = {self.dlogz}). Consider a looser dlogz, more "
                    f"num_live points, or more num_inner_steps.")
            run_key, subkey = jax.random.split(run_key)
            state, step_info = step(subkey, state)
            dead.append(step_info)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/inference/ -v`
Expected: all PASS (11 in `test_nested.py` + existing `test_mclmc.py`, `test_smc.py`, `test_target.py`).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py tests/inference/test_nested.py
git commit -m "feat(inference): NestedSampler max_iterations guard + bimodal/pipeline tests"
```
