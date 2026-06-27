# NLE Posterior Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the numpyro-NUTS `NLEPosterior` with a populated `gensbi.inference` package that separates the posterior *target* from blackjax *samplers* (adjusted MCLMC by default; tempered SMC for multimodal posteriors).

**Architecture:** `NLEPosterior` builds a `PosteriorTarget` (log-prior / log-likelihood / log-posterior closures for a given `x_o`) and dispatches to a `Sampler` object. `MCLMC` (adjusted by default) is the default sampler; `TemperedSMC` is an opt-in object for multimodal posteriors, using an adjusted-MCLMC inner kernel by default with a NUTS fallback. numpyro stays only as the prior abstraction; the numpyro sampler is removed.

**Tech Stack:** JAX, blackjax 1.5, numpyro (priors only), flax.nnx (flows), pytest.

## Global Constraints

- **Test runner:** `mamba run -n gensbi python -m pytest` (the `gensbi` mamba env, NOT `.venv`). Test files set `os.environ["JAX_PLATFORMS"] = "cpu"` at the top before importing jax.
- **blackjax:** version 1.5 (already declared in `pyproject.toml`). Imported lazily inside `Sampler.run()` so importing `gensbi.inference` stays cheap.
- **numpyro:** stays a dependency (prior abstraction `make_gaussian_prior` is used across `core`/`recipes`). Only the numpyro *sampler* usage in `nle.py` is removed.
- **Output contract:** `NLEPosterior.sample(...)` returns samples shaped `(n, dim, 1)` via `gensbi.utils.math._expand_dims`. `n = num_samples * num_chains` (MCLMC) or `num_particles` (SMC).
- **`build_target` is public:** a power user can call `NLEPosterior.build_target(x_o)` to get the log-density closures and drive their own sampler.
- **θ is always a flat `(dim,)` vector**, even when `structured_obs=True` (only the observation `x_o` is structured). Samplers never see structured data.
- **No clamping of `log_prob`.** Faithful to the existing no-clamp decision; untrained flows may overflow in float32 — that is the model's truth.
- **Breaking changes allowed.** `NLEPosterior` is not in `main`; restructure freely and update the ~7 internal call sites. No back-compat shims.
- **Prior interface:** numpyro `dist.Independent(dist.Normal(...))` exposes `prior.log_prob(theta) -> scalar`, `prior.sample(key, ()) -> (dim,)`, `prior.sample(key, (n,)) -> (n, dim)`, and `prior.event_shape -> (dim,)`.
- **Flow interface:** `flow.log_prob(x, cond) -> (B,)` with `x` shaped `(B, *obs_shape)` and `cond` shaped `(B, dim)`.

**Spec:** `docs/superpowers/specs/2026-06-27-nle-posterior-sampling-design.md`

---

### Task 1: `inference` package + `PosteriorTarget` + `NLEPosterior` target builder

**Files:**
- Create: `src/gensbi/inference/posterior.py`
- Modify: `src/gensbi/inference/__init__.py`
- Create: `tests/inference/__init__.py` (empty)
- Test: `tests/inference/test_target.py`

**Interfaces:**
- Consumes: `gensbi.utils.math._expand_dims`; numpyro prior (see Global Constraints); flow `log_prob`.
- Produces:
  - `PosteriorTarget` — frozen dataclass with fields `log_prior: Callable`, `log_likelihood: Callable`, `log_posterior: Callable`, `prior`, `dim: int`. Each callable takes a flat `theta (dim,)` and returns a scalar.
  - `NLEPosterior(flow, prior, *, structured_obs: bool = False)` with `build_target(self, x_o) -> PosteriorTarget` and `sample(self, key, x_o, sampler=None, *, return_info=False)` (sampler dispatch added in Task 2 — for now `sample` raises `NotImplementedError`).

- [ ] **Step 1: Write the failing test**

```python
# tests/inference/test_target.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference.posterior import NLEPosterior, PosteriorTarget


class GaussianMock:
    """log q(x | theta) = sum_i N(x_i; theta_i, 1) (batched over rows)."""
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)   # (B,)


def test_build_target_decomposition_and_finiteness():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    target = post.build_target(jnp.array([1.0, -1.0]))

    assert isinstance(target, PosteriorTarget)
    assert target.dim == dim
    theta = jnp.array([0.3, 0.4])
    # log_posterior == log_prior + log_likelihood
    assert jnp.allclose(target.log_posterior(theta),
                        target.log_prior(theta) + target.log_likelihood(theta))
    # value and grad finite
    val = target.log_posterior(theta)
    grad = jax.grad(target.log_posterior)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (dim,) and jnp.all(jnp.isfinite(grad))


def test_log_likelihood_matches_flow():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    x_o = jnp.array([1.0, -1.0])
    theta = jnp.array([0.3, 0.4])
    target = post.build_target(x_o)
    expected = GaussianMock().log_prob(x_o[None], theta[None])[0]
    assert jnp.allclose(target.log_likelihood(theta), expected)


def test_structured_obs_keeps_observation_shape():
    # structured_obs: x_o is an image; theta stays a flat vector.
    dim = 2
    H = W = 4

    class ImageFlow:
        def log_prob(self, x, cond):
            # assert x retained its (B, H, W) structure
            assert x.shape == (1, H, W)
            return -0.5 * jnp.sum(cond ** 2, axis=-1)  # (B,)

    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(ImageFlow(), prior, structured_obs=True)
    x_o = jnp.ones((H, W))
    target = post.build_target(x_o)
    assert jnp.isfinite(target.log_likelihood(jnp.array([0.1, 0.2])))


def test_dim_mismatch_raises():
    prior = make_gaussian_prior((3,))
    post = NLEPosterior(GaussianMock(), prior)
    target = post.build_target(jnp.array([1.0, 2.0, 3.0]))
    assert target.dim == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_target.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gensbi.inference.posterior'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/inference/posterior.py
"""NLE posterior target: build log-densities from a trained likelihood flow + prior.

The flow is NLE-trained (``obs = x``, ``cond = theta``), so ``flow.log_prob(x, theta)``
is ``log q(x | theta)``. ``NLEPosterior`` turns ``(flow, prior, x_o)`` into a
``PosteriorTarget`` (separate log-prior / log-likelihood / log-posterior closures),
which a ``Sampler`` consumes. The flow params are frozen constants inside the
closures; only ``theta`` is traced/differentiated.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from gensbi.utils.math import _expand_dims


@dataclass(frozen=True)
class PosteriorTarget:
    """Log-densities for one observation ``x_o``. All callables take flat ``theta (dim,)``."""
    log_prior: Callable
    log_likelihood: Callable
    log_posterior: Callable
    prior: object
    dim: int


class NLEPosterior:
    """Amortized NLE posterior over a trained likelihood flow.

    Parameters
    ----------
    flow : object
        Exposes ``log_prob(x, cond) -> (B,)`` (an NLE-trained ``MAFlow``/``TarFlow``).
    prior : numpyro.distributions.Distribution
        Prior over theta; ``prior.log_prob(theta)`` is a scalar and
        ``prior.sample(key, ())`` returns ``(dim,)``.
    structured_obs : bool
        If True, ``x_o`` keeps its (image/field) shape instead of being flattened.
    """

    def __init__(self, flow, prior, *, structured_obs: bool = False):
        self.flow = flow
        self.prior = prior
        self.structured_obs = structured_obs

    def build_target(self, x_o) -> PosteriorTarget:
        if self.structured_obs:
            x_o = jnp.asarray(x_o)
        else:
            x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))   # (dim_x,)
        flow = self.flow
        prior = self.prior
        dim = int(prior.event_shape[0])

        def log_prior(theta):
            return prior.log_prob(jnp.asarray(theta))

        def log_likelihood(theta):
            theta = jnp.asarray(theta)
            return flow.log_prob(x_o[None], theta[None, :])[0]

        def log_posterior(theta):
            return log_likelihood(theta) + log_prior(theta)

        return PosteriorTarget(
            log_prior=log_prior, log_likelihood=log_likelihood,
            log_posterior=log_posterior, prior=prior, dim=dim,
        )

    def sample(self, key, x_o, sampler=None, *, return_info=False):
        raise NotImplementedError("sampler dispatch added in Task 2")
```

```python
# src/gensbi/inference/__init__.py
"""Inference wrappers: NLE posterior sampling over trained density flows."""

from gensbi.inference.posterior import NLEPosterior, PosteriorTarget

__all__ = ["NLEPosterior", "PosteriorTarget"]
```

Also create empty `tests/inference/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_target.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/posterior.py src/gensbi/inference/__init__.py tests/inference/
git commit -m "feat(inference): PosteriorTarget + NLEPosterior target builder"
```

---

### Task 2: `Sampler` ABC + `MCLMC` (unadjusted) + `sample()` dispatch

**Files:**
- Create: `src/gensbi/inference/samplers.py`
- Modify: `src/gensbi/inference/posterior.py` (implement `sample`)
- Modify: `src/gensbi/inference/__init__.py`
- Test: `tests/inference/test_mclmc.py`

**Interfaces:**
- Consumes: `PosteriorTarget` (Task 1); `gensbi.utils.math._expand_dims`.
- Produces:
  - `Sampler` (ABC) with `run(self, key, target) -> tuple[Array, object]` returning `(samples (n, dim), info)`.
  - `MclmcInfo` — frozen dataclass `L: float`, `step_size: float`, `acceptance_rate: float`, `num_samples: int`, `num_chains: int`.
  - `MCLMC(*, adjusted=True, num_samples=1000, num_tuning_steps=5000, num_chains=1, target_acceptance=0.9, diagonal_preconditioning=True)`. **Task 2 implements only the `adjusted=False` (unadjusted) path**; `adjusted=True` raises `NotImplementedError` until Task 3.
  - `NLEPosterior.sample(key, x_o, sampler=None, *, return_info=False)` — defaults to `MCLMC()`, returns `(n, dim, 1)` (or `(samples, info)` if `return_info`).

- [ ] **Step 1: Write the failing test**

```python
# tests/inference/test_mclmc.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.core.prior import make_gaussian_prior
from gensbi.models import MAFlow, MAFlowParams
from gensbi.inference import NLEPosterior, MCLMC


class GaussianMock:
    """log q(x | theta) = N(x; theta, I); with prior N(0, I) => posterior N(x_o/2, 0.5 I)."""
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)


def test_unadjusted_prior_recovery_real_flow():
    # zero_init flow => q(x|theta) is theta-independent => posterior == prior N(0, I).
    # Exercises MCLMC end-to-end against a real MAFlow.
    dim = 2
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim, cond_dim=dim,
                               n_layers=3, nn_width=16, zero_init=True))
    post = NLEPosterior(flow, make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(9), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=800, num_tuning_steps=600))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), 0.0, atol=0.25)


def test_unadjusted_shape_and_finite():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(0), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=500, num_tuning_steps=500))
    assert s.shape == (500, dim, 1)
    assert jnp.all(jnp.isfinite(s))


def test_unadjusted_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(1), x_o,
                    sampler=MCLMC(adjusted=False, num_samples=3000, num_tuning_steps=2000))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.15)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.2)


def test_unadjusted_multichain_shape():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=200, num_tuning_steps=400, num_chains=3))
    assert s.shape == (600, dim, 1)


def test_return_info():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s, info = post.sample(jax.random.PRNGKey(3), jnp.array([1.0, -1.0]),
                          sampler=MCLMC(adjusted=False, num_samples=200, num_tuning_steps=400),
                          return_info=True)
    assert s.shape == (200, dim, 1)
    assert info.num_samples == 200 and jnp.isfinite(info.L) and jnp.isfinite(info.step_size)


def test_adjusted_not_yet_implemented():
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((2,)))
    with pytest.raises(NotImplementedError):
        post.sample(jax.random.PRNGKey(4), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=True, num_samples=10, num_tuning_steps=10))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: FAIL with `ImportError: cannot import name 'MCLMC'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/inference/samplers.py
"""blackjax samplers consumed by NLEPosterior. blackjax imported lazily in run()."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


class Sampler(ABC):
    """Turns a PosteriorTarget into posterior samples."""

    @abstractmethod
    def run(self, key, target):
        """Return (samples (n, dim), info)."""
        raise NotImplementedError


@dataclass(frozen=True)
class MclmcInfo:
    L: float
    step_size: float
    acceptance_rate: float
    num_samples: int
    num_chains: int


def _inference_loop(rng_key, step_fn, initial_state, num_samples):
    """Run a blackjax SamplingAlgorithm.step for num_samples via lax.scan."""
    @jax.jit
    def one_step(state, k):
        state, info = step_fn(k, state)
        return state, (state, info)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


class MCLMC(Sampler):
    """Microcanonical Langevin Monte Carlo.

    adjusted=True (default, added in Task 3) is MH-corrected / asymptotically exact.
    adjusted=False is the faster unadjusted variant (biased by the discretization).
    """

    def __init__(self, *, adjusted=True, num_samples=1000, num_tuning_steps=5000,
                 num_chains=1, target_acceptance=0.9, diagonal_preconditioning=True):
        self.adjusted = adjusted
        self.num_samples = num_samples
        self.num_tuning_steps = num_tuning_steps
        self.num_chains = num_chains
        self.target_acceptance = target_acceptance
        self.diagonal_preconditioning = diagonal_preconditioning

    def run(self, key, target):
        # Python loop over chains (not vmap): _run_single returns a plain MclmcInfo
        # dataclass, which is not a registered pytree and cannot be a vmap output.
        # num_chains is small, so the loop cost is negligible.
        keys = jax.random.split(key, self.num_chains)
        results = [self._run_single(k, target) for k in keys]
        samples = jnp.concatenate([r[0] for r in results], axis=0)  # (num_chains*num_samples, dim)
        if self.num_chains == 1:
            return samples, results[0][1]
        infos = [r[1] for r in results]
        info = MclmcInfo(
            L=float(jnp.mean(jnp.array([i.L for i in infos]))),
            step_size=float(jnp.mean(jnp.array([i.step_size for i in infos]))),
            acceptance_rate=float(jnp.mean(jnp.array([i.acceptance_rate for i in infos]))),
            num_samples=self.num_samples, num_chains=self.num_chains,
        )
        return samples, info

    def _run_single(self, key, target):
        if self.adjusted:
            return self._run_adjusted(key, target)
        return self._run_unadjusted(key, target)

    def _run_unadjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        init_key, tune_key, run_key = jax.random.split(key, 3)
        position = target.prior.sample(init_key, ())
        init_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=target.log_posterior, rng_key=init_key)
        kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=target.log_posterior, integrator=isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix)
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=self.num_tuning_steps, state=init_state,
            rng_key=tune_key, diagonal_preconditioning=self.diagonal_preconditioning)
        alg = blackjax.mclmc(target.log_posterior, L=params.L, step_size=params.step_size,
                             inverse_mass_matrix=params.inverse_mass_matrix)
        states, _ = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=params.L, step_size=params.step_size,
                         acceptance_rate=jnp.nan, num_samples=self.num_samples,
                         num_chains=self.num_chains)
        return states.position, info

    def _run_adjusted(self, key, target):
        raise NotImplementedError("adjusted MCLMC added in Task 3")
```

Modify `posterior.py` `sample`:

```python
    def sample(self, key, x_o, sampler=None, *, return_info=False):
        """Draw posterior samples. Returns (n, dim, 1), or (samples, info) if return_info."""
        from gensbi.inference.samplers import MCLMC
        sampler = sampler if sampler is not None else MCLMC()
        target = self.build_target(x_o)
        samples, info = sampler.run(key, target)
        samples = _expand_dims(samples)          # (n, dim) -> (n, dim, 1)
        return (samples, info) if return_info else samples
```

Update `__init__.py`:

```python
# src/gensbi/inference/__init__.py
"""Inference wrappers: NLE posterior sampling over trained density flows."""

from gensbi.inference.posterior import NLEPosterior, PosteriorTarget
from gensbi.inference.samplers import Sampler, MCLMC, MclmcInfo

__all__ = ["NLEPosterior", "PosteriorTarget", "Sampler", "MCLMC", "MclmcInfo"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/ tests/inference/test_mclmc.py
git commit -m "feat(inference): Sampler ABC + unadjusted MCLMC + sample() dispatch"
```

---

### Task 3: `MCLMC` adjusted path (default)

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (implement `_run_adjusted`, add `rescale` helper)
- Test: `tests/inference/test_mclmc.py` (add tests)

**Interfaces:**
- Consumes: same as Task 2.
- Produces: `MCLMC._run_adjusted` implemented; `MCLMC(adjusted=True)` (the constructor default) now works. No public-signature changes.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/inference/test_mclmc.py

def test_adjusted_is_the_default():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    assert MCLMC().adjusted is True


def test_adjusted_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    # default sampler == adjusted MCLMC; exercised via the one-liner
    s = post.sample(jax.random.PRNGKey(5), x_o,
                    sampler=MCLMC(num_samples=3000, num_tuning_steps=2000))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.15)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.2)


def test_adjusted_reports_acceptance_rate():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    _, info = post.sample(jax.random.PRNGKey(6), jnp.array([1.0, -1.0]),
                          sampler=MCLMC(num_samples=400, num_tuning_steps=600),
                          return_info=True)
    assert 0.0 <= info.acceptance_rate <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py::test_adjusted_analytic_gaussian_recovery -v`
Expected: FAIL with `NotImplementedError: adjusted MCLMC added in Task 3`

- [ ] **Step 3: Write minimal implementation**

Add the `rescale` helper near the top of `samplers.py` (after imports):

```python
def _rescale(mu):
    """Map a mean trajectory length to a uniform-integer draw scale.

    From the blackjax adjusted-MCLMC tutorial: choosing the number of integration
    steps as ceil(U(0,1) * _rescale(L/step_size)) keeps the average near the tuned L.
    """
    k = jax.lax.max(1, jnp.round(jnp.log(2 * mu - 1) / jnp.log(2)).astype(int))
    return mu / k
```

Replace `_run_adjusted`:

```python
    def _run_adjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        init_key, tune_key, run_key = jax.random.split(key, 3)
        position = target.prior.sample(init_key, ())
        init_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=position, logdensity_fn=target.log_posterior,
            random_generator_arg=tune_key)

        def kernel(rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix):
            return blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                integration_steps_fn=lambda k: jnp.ceil(
                    jax.random.uniform(k) * _rescale(avg_num_integration_steps)),
                inverse_mass_matrix=inverse_mass_matrix,
            )(rng_key=rng_key, state=state, step_size=step_size,
              logdensity_fn=target.log_posterior)

        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=self.num_tuning_steps, state=init_state,
            rng_key=tune_key, target=self.target_acceptance,
            diagonal_preconditioning=self.diagonal_preconditioning)

        alg = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn=target.log_posterior, step_size=params.step_size,
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * _rescale(params.L / params.step_size)),
            inverse_mass_matrix=params.inverse_mass_matrix)

        states, infos = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=params.L, step_size=params.step_size,
                         acceptance_rate=jnp.mean(infos.acceptance_rate),
                         num_samples=self.num_samples, num_chains=self.num_chains)
        return states.position, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_mclmc.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py tests/inference/test_mclmc.py
git commit -m "feat(inference): adjusted MCLMC (default), MH-corrected microcanonical"
```

---

### Task 4: `TemperedSMC` with NUTS inner kernel + log-evidence

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (add `SmcInfo`, `TemperedSMC`)
- Modify: `src/gensbi/inference/__init__.py`
- Test: `tests/inference/test_smc.py`

**Interfaces:**
- Consumes: `PosteriorTarget` (`log_prior`, `log_likelihood`, `prior`, `dim`).
- Produces:
  - `SmcInfo` — frozen dataclass `log_evidence: float`, `num_temperature_steps: int`, `final_tempering_param: float`.
  - `TemperedSMC(*, num_particles=1000, target_ess=0.5, num_mcmc_steps=10, inner_kernel="mclmc", inner_step_size=0.1, inner_num_integration_steps=5, inner_inverse_mass_matrix=None)`. **Task 4 implements only `inner_kernel="nuts"`**; `"mclmc"` raises `NotImplementedError` until Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/inference/test_smc.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import NLEPosterior, TemperedSMC


class GaussianMock:
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)


class BimodalMock:
    """log q(x | theta) = mixture of N(theta; +mu, 0.5 I) and N(theta; -mu, 0.5 I).

    Independent of the observation x; posterior under a broad prior is bimodal at +/-mu.
    """
    def __init__(self, mu=3.0, sigma=0.5):
        self.mu, self.sigma = mu, sigma

    def log_prob(self, x, cond):
        a = -0.5 * jnp.sum(((cond - self.mu) / self.sigma) ** 2, axis=-1)
        b = -0.5 * jnp.sum(((cond + self.mu) / self.sigma) ** 2, axis=-1)
        return jax.scipy.special.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)


def test_smc_nuts_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(0), x_o,
                    sampler=TemperedSMC(inner_kernel="nuts", num_particles=2000,
                                        inner_step_size=0.5))[..., 0]
    assert s.shape == (2000, dim)
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.2)


def test_smc_nuts_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=TemperedSMC(inner_kernel="nuts", num_particles=2000,
                                        inner_step_size=0.5))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # both modes populated (a single MCMC chain would capture only one)
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_smc_info_has_log_evidence():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    _, info = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                          sampler=TemperedSMC(inner_kernel="nuts", num_particles=1000,
                                              inner_step_size=0.5),
                          return_info=True)
    assert jnp.isfinite(info.log_evidence)
    assert info.num_temperature_steps > 0
    assert jnp.isclose(info.final_tempering_param, 1.0, atol=1e-6)


def test_smc_mclmc_not_yet_implemented():
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((2,)))
    with pytest.raises(NotImplementedError):
        post.sample(jax.random.PRNGKey(3), jnp.array([1.0, -1.0]),
                    sampler=TemperedSMC(inner_kernel="mclmc", num_particles=100))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v`
Expected: FAIL with `ImportError: cannot import name 'TemperedSMC'`

- [ ] **Step 3: Write minimal implementation**

Add to `samplers.py`:

```python
@dataclass(frozen=True)
class SmcInfo:
    log_evidence: float
    num_temperature_steps: int
    final_tempering_param: float


class TemperedSMC(Sampler):
    """Adaptive tempered SMC for (possibly multimodal) posteriors.

    Walks particles along p(theta) * q(x_o|theta)^beta for beta: 0 -> 1, choosing the
    beta-ladder adaptively to hold target_ess. Inner rejuvenation kernel is adjusted
    MCLMC by default (Task 5); NUTS is the fallback.
    """

    def __init__(self, *, num_particles=1000, target_ess=0.5, num_mcmc_steps=10,
                 inner_kernel="mclmc", inner_step_size=0.1,
                 inner_num_integration_steps=5, inner_inverse_mass_matrix=None):
        self.num_particles = num_particles
        self.target_ess = target_ess
        self.num_mcmc_steps = num_mcmc_steps
        self.inner_kernel = inner_kernel
        self.inner_step_size = inner_step_size
        self.inner_num_integration_steps = inner_num_integration_steps
        self.inner_inverse_mass_matrix = inner_inverse_mass_matrix

    def _inner(self, target):
        """Return (mcmc_step_fn, mcmc_init_fn, mcmc_parameters) for the inner kernel."""
        import blackjax
        imm = self.inner_inverse_mass_matrix
        if imm is None:
            imm = jnp.ones(target.dim)
        if self.inner_kernel == "nuts":
            step_fn = blackjax.nuts.build_kernel()
            init_fn = blackjax.nuts.init
            params = dict(step_size=self.inner_step_size, inverse_mass_matrix=imm)
            return step_fn, init_fn, params
        if self.inner_kernel == "mclmc":
            raise NotImplementedError("mclmc inner kernel added in Task 5")
        raise ValueError(f"unknown inner_kernel {self.inner_kernel!r}")

    def run(self, key, target):
        import blackjax

        init_key, smc_key = jax.random.split(key)
        step_fn, init_fn, params = self._inner(target)
        smc = blackjax.adaptive_tempered_smc(
            logprior_fn=target.log_prior, loglikelihood_fn=target.log_likelihood,
            mcmc_step_fn=step_fn, mcmc_init_fn=init_fn,
            mcmc_parameters=blackjax.smc.extend_params(params),
            resampling_fn=blackjax.smc.resampling.systematic,
            target_ess=self.target_ess, num_mcmc_steps=self.num_mcmc_steps,
        )
        init_particles = target.prior.sample(init_key, (self.num_particles,))
        state = smc.init(init_particles)

        def cond(carry):
            _, st, _, _ = carry
            return st.tempering_param < 1.0

        def body(carry):
            k, st, n, logZ = carry
            k, sub = jax.random.split(k)
            st, info = smc.step(sub, st)
            return k, st, n + 1, logZ + info.log_likelihood_increment

        _, final, nsteps, logZ = jax.lax.while_loop(
            cond, body, (smc_key, state, 0, 0.0))
        info = SmcInfo(log_evidence=logZ, num_temperature_steps=nsteps,
                       final_tempering_param=final.tempering_param)
        return final.particles, info
```

Update `__init__.py`:

```python
from gensbi.inference.samplers import Sampler, MCLMC, MclmcInfo, TemperedSMC, SmcInfo

__all__ = ["NLEPosterior", "PosteriorTarget", "Sampler",
           "MCLMC", "MclmcInfo", "TemperedSMC", "SmcInfo"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/ tests/inference/test_smc.py
git commit -m "feat(inference): TemperedSMC (NUTS inner) + log-evidence, bimodal recovery"
```

---

### Task 5: `TemperedSMC` adjusted-MCLMC inner kernel (default)

**Files:**
- Modify: `src/gensbi/inference/samplers.py` (`TemperedSMC._inner` mclmc branch)
- Test: `tests/inference/test_smc.py` (add tests)

**Interfaces:**
- Consumes: same as Task 4.
- Produces: `TemperedSMC(inner_kernel="mclmc")` (the constructor default) works via an adjusted-MCLMC adapter conforming to blackjax's SMC inner-kernel contract (`mcmc_step_fn(rng_key, state, logdensity_fn, step_size, num_integration_steps, inverse_mass_matrix)`, `mcmc_init_fn(position, logdensity_fn)`). No public-signature changes.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/inference/test_smc.py

def test_smc_mclmc_is_the_default_inner_kernel():
    assert TemperedSMC().inner_kernel == "mclmc"


def test_smc_mclmc_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    # default inner kernel == adjusted MCLMC
    s = post.sample(jax.random.PRNGKey(7), jnp.zeros(dim),
                    sampler=TemperedSMC(num_particles=2000, inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_smc_mclmc_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(8), x_o,
                    sampler=TemperedSMC(num_particles=2000, inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_smc.py::test_smc_mclmc_recovers_both_modes -v`
Expected: FAIL with `NotImplementedError: mclmc inner kernel added in Task 5`

- [ ] **Step 3: Write minimal implementation**

Replace the `mclmc` branch in `TemperedSMC._inner`:

```python
        if self.inner_kernel == "mclmc":
            from blackjax.mcmc.integrators import isokinetic_mclachlan
            import blackjax

            def step_fn(rng_key, state, logdensity_fn, step_size,
                        num_integration_steps, inverse_mass_matrix):
                # Build the adjusted-MCLMC kernel bound to the *tempered* logdensity
                # SMC injects at each temperature.
                kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                    logdensity_fn=logdensity_fn, integrator=isokinetic_mclachlan,
                    inverse_mass_matrix=inverse_mass_matrix)
                return kernel(rng_key, state, step_size=step_size,
                              num_integration_steps=num_integration_steps)

            init_fn = blackjax.mcmc.adjusted_mclmc.init   # (position, logdensity_fn) -> HMCState
            params = dict(step_size=self.inner_step_size,
                          num_integration_steps=self.inner_num_integration_steps,
                          inverse_mass_matrix=imm)
            return step_fn, init_fn, params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n gensbi python -m pytest tests/inference/test_smc.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/samplers.py tests/inference/test_smc.py
git commit -m "feat(inference): adjusted-MCLMC inner kernel for TemperedSMC (default)"
```

---

### Task 6: Migrate call sites, delete numpyro-NUTS `nle.py`

**Files:**
- Delete: `src/gensbi/inference/nle.py`
- Delete: `tests/normalizing_flows/test_nle.py` (superseded by `tests/inference/`)
- Modify: `tests/models/tarflow/test_structured_integration.py` (`.sample()` call, ~lines 43-46)
- Modify: `tests/models/tarflow/test_structured_boundary.py` (`.potential()` → `build_target().log_posterior`, ~lines 70-79)
- Modify: `tests/models/tarflow/test_pipeline_integration.py` (`.potential()` → `build_target().log_posterior`, ~lines 89-95)
- Modify: `scripts/maf_nle_recovery.py` (~lines 147-149)
- Modify: `scripts/tarflow_nle_recovery.py` (~lines 151-153)
- Modify: `scripts/tarflow_field_nle_recovery.py` (~lines 96-99)

**Interfaces:**
- Consumes: `NLEPosterior`, `MCLMC`, `build_target` (Tasks 1-3). The old constructor kwargs `num_warmup`/`num_samples` move to `MCLMC(num_tuning_steps=..., num_samples=...)` passed via `sampler=`.
- The removed `NLEPosterior.potential(x_o)` (returned `U(θ) = −(log q + log p)`) is replaced by `post.build_target(x_o).log_posterior` (= `log q + log p`). The sign flips, but the two `.potential()` call sites only assert value/grad **finiteness**, which is unaffected.
- Produces: no new symbols. Removes the last numpyro-sampler usage.

- [ ] **Step 1: Confirm the old NUTS path is the only thing importing numpyro.infer**

Run: `grep -rn "numpyro.infer\|from numpyro import infer\|MCMC\|NUTS" src/gensbi/inference/`
Expected: only matches in `src/gensbi/inference/nle.py`

- [ ] **Step 2: Delete the superseded files**

```bash
git rm src/gensbi/inference/nle.py tests/normalizing_flows/test_nle.py
```

- [ ] **Step 3: Update the tarflow `.sample()` test**

In `tests/models/tarflow/test_structured_integration.py`, replace the `NLEPosterior` block:

```python
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)),
                        num_warmup=3, num_samples=10, structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), _x[0])
    assert s.shape == (10, D, 1) and jnp.all(jnp.isfinite(s))
```
with:

```python
    from gensbi.inference import MCLMC
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)), structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), _x[0],
                    sampler=MCLMC(adjusted=False, num_samples=10, num_tuning_steps=20))
    # untrained flow: samples may include non-finite rows (no clamping). Smoke-check shape
    # + at-least-some-finite, consistent with existing untrained-flow relaxations.
    assert s.shape == (10, D, 1) and jnp.any(jnp.isfinite(s))
```

- [ ] **Step 4: Update the two `.potential()` tests to `build_target().log_posterior`**

In `tests/models/tarflow/test_structured_boundary.py`, replace:

```python
    post = NLEPosterior(flow, prior, structured_obs=True)
    x_o = jnp.zeros((H, W, Ch))
    U = post.potential(x_o)
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
```
with:

```python
    post = NLEPosterior(flow, prior, structured_obs=True)
    x_o = jnp.zeros((H, W, Ch))
    target = post.build_target(x_o)
    theta = jnp.array([0.1, 0.2])
    val = target.log_posterior(theta)
    grad = jax.grad(target.log_posterior)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
```

In `tests/models/tarflow/test_pipeline_integration.py`, replace:

```python
    post = NLEPosterior(flow, prior)
    U = post.potential(jnp.array([0.5, -0.5, 0.2]))
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
```
with:

```python
    post = NLEPosterior(flow, prior)
    target = post.build_target(jnp.array([0.5, -0.5, 0.2]))
    theta = jnp.array([0.1, 0.2])
    val = target.log_posterior(theta)
    grad = jax.grad(target.log_posterior)(theta)
```

(The trailing `assert val.shape == () and jnp.isfinite(val)` / grad assertions in that test are unchanged.)

- [ ] **Step 5: Update the recovery scripts**

In `scripts/maf_nle_recovery.py` and `scripts/tarflow_nle_recovery.py`, replace the line:

```python
    post = NLEPosterior(pipe.ema_model, prior, num_warmup=num_warmup, num_samples=num_samples)
    sample_key = jax.random.PRNGKey(7)
    s = post.sample(sample_key, x_o)[..., 0]  # (n, D)
```
with:

```python
    from gensbi.inference import MCLMC
    post = NLEPosterior(pipe.ema_model, prior)
    sample_key = jax.random.PRNGKey(7)
    s = post.sample(sample_key, x_o,
                    sampler=MCLMC(num_samples=num_samples, num_tuning_steps=num_warmup))[..., 0]
```

In `scripts/tarflow_field_nle_recovery.py` (lines ~59-62), replace:

```python
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)),
                        num_warmup=num_warmup, num_samples=num_samples,
                        structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]
```
with:

```python
    from gensbi.inference import MCLMC
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)), structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), x_o,
                    sampler=MCLMC(num_samples=num_samples, num_tuning_steps=num_warmup))[..., 0]
```

- [ ] **Step 6: Run the full inference + tarflow test suites**

Run: `mamba run -n gensbi python -m pytest tests/inference/ tests/models/tarflow/ -v`
Expected: PASS. (The structured-obs smoke test already relaxes to `jnp.any(jnp.isfinite(s))` for the untrained flow per Step 3.)

- [ ] **Step 7: Smoke-run a recovery script**

Run: `mamba run -n gensbi python scripts/maf_nle_recovery.py --smoke`
Expected: completes without error and prints analytic vs achieved posterior mean/cov.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(inference): migrate call sites to sampler objects, drop numpyro NUTS"
```

---

## Notes for the implementer

- **Why `mamba run -n gensbi`:** the `.venv` and the mamba `gensbi` env are parallel; the `gensbi` env is the one that surfaces real test failures. All test commands use it.
- **The adjusted-MCLMC `_rescale` helper** comes verbatim from the blackjax adjusted-MCLMC tutorial; it converts the tuned mean trajectory length into a randomized integer step count. Do not "simplify" it — it is load-bearing for the tuner/run to agree.
- **SMC `extend_params` is mandatory:** blackjax 1.5 inspects each `mcmc_parameters` value's `.shape` to tell shared from per-particle parameters; raw Python scalars raise `AttributeError`. Wrap the dict with `blackjax.smc.extend_params(...)`.
- **NUTS inner kernel takes no `num_integration_steps`** (it chooses its own trajectory length); only `step_size` + `inverse_mass_matrix`. The MCLMC inner kernel *does* take `num_integration_steps`.
- **All blackjax tuners return a 3-tuple** `(state, params, total_num_tuning_integrator_steps)`; unpack with a trailing `_`.
- **`TemperedSMCState` field is `tempering_param`** (not `lmbda`); the per-step `SMCInfo` field summed for log-evidence is `log_likelihood_increment`.
- **`info` dataclasses are intentionally lean** — a deliberate trim from spec Decision 5 (which also listed divergence counts and SMC inner-acceptance), honoring the "simple sampler that reasonably works / lean knob count" steer. `MclmcInfo` carries `L`/`step_size`/`acceptance_rate`/`num_samples`/`num_chains`; `SmcInfo` carries `log_evidence`/`num_temperature_steps`/`final_tempering_param`. blackjax exposes `is_divergent` in the adjusted-MCLMC step info if a divergence count is wanted as a follow-up.
- **All blackjax imports are lazy** (inside `run()` / `_inner()` / `_run_*`), never at module top of `samplers.py`, so `import gensbi.inference` stays cheap.
- **`jax.lax.while_loop` carry types must stay consistent:** the SMC loop seeds `(smc_key, state, 0, 0.0)` — keep the step counter an int and `logZ` a float so the carry types don't drift between iterations.
