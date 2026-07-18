# Nested Sampling for NLE Posterior Inference — Design

**Date:** 2026-07-11
**Branch:** `nested-sampling`
**Status:** Approved

## Goal

Add nested sampling (blackjax ≥ 1.6 `nss`, Nested Slice Sampling) as an
opt-in alternative to MCMC-based sampling in `gensbi.inference`, wired into
the existing NLE pipeline for normalizing flows. Motivation is twofold and
equal: robust posterior samples for multimodal/degenerate targets, and
log-evidence estimates for Bayesian model comparison between NLE models.

## Decisions (from brainstorming)

- **Opt-in**: adjusted MCLMC remains the default in `NLEPosterior.sample()`;
  nested sampling is selected via `sampler=NestedSampler(...)`.
- **Raw run retained**: the info object carries the finalised blackjax
  dead-point object so anesthetic support can be added later without re-runs.
  Anesthetic integration itself is **out of scope** now.
- **Scope**: sampler + unit tests only. No recovery-script wiring, no docs
  page in this iteration.
- **Loop style**: blackjax-canonical Python `while` loop over a jitted
  `step` (dead points accumulate with variable length). A fully-jitted
  `lax.while_loop` with a preallocated buffer was rejected as needless
  complexity at SBI dimensionalities.

## API surface

Two additions to `src/gensbi/inference/samplers.py`, exported from
`gensbi.inference.__init__`:

```python
class NestedSampler(Sampler):
    def __init__(self, *, num_live=500, num_delete=None, num_inner_steps=None,
                 num_samples=1000, dlogz=-3.0, max_iterations=100_000):
```

- `num_live` — number of live points (default 500).
- `num_delete` — particles deleted/replaced per step; `None` →
  `num_live // 10` (device batching amortizes the Python-loop overhead).
- `num_inner_steps` — constrained slice moves per replacement; `None` →
  `max(5, 2 * target.dim)` resolved inside `run()` (blackjax's rule of
  thumb; `dim` comes from the target so users never pass it).
- `num_samples` — equal-weight posterior draws returned (default 1000,
  matching `MCLMC`).
- `dlogz` — termination threshold, blackjax convention: stop when
  `logZ_live − logZ < dlogz` (default −3.0; use −10 near phase transitions).
- `max_iterations` — safety cap on the outer loop (default 100 000).

```python
@dataclass(frozen=True)
class NestedSamplerInfo:
    log_evidence: float        # mean over stochastic logZ draws
    log_evidence_err: float    # std over those draws
    ess: float                 # effective sample size of the weighted run
    num_dead: int              # total dead points accumulated
    dead: object               # raw finalised blackjax NSInfo (future anesthetic use)
```

Usage (unchanged pipeline pattern):

```python
post = NLEPosterior(flow, prior)
samples, info = post.sample(key, x_o, sampler=NestedSampler(), return_info=True)
```

## Data flow in `run(key, target)`

1. Split `key` into init / run / weights / resample keys.
2. `particles = target.prior.sample(init_key, (num_live,))` — live points
   start from the prior, as NS requires.
3. `algo = blackjax.nss(logprior_fn=target.log_prior,
   loglikelihood_fn=target.log_likelihood, num_inner_steps=...,
   num_delete=...)`. The `PosteriorTarget` prior/likelihood split maps 1:1
   onto the NS API — **no changes to `posterior.py`**.
4. `state = algo.init(particles)`; run `jax.jit(algo.step)` in a Python
   `while` loop, appending each step's dead-point info; stop when
   `state.integrator.logZ_live - state.integrator.logZ < dlogz`.
5. `ns_run = blackjax.ns.utils.finalise(state, dead)`.
   Evidence: `log_weights(weights_key, ns_run, shape=100)` →
   `logsumexp` over the dead axis → mean/std across the 100 stochastic
   volume draws. ESS via `blackjax.ns.utils.ess`.
6. Equal-weight draws via `blackjax.ns.utils.sample(resample_key, ns_run,
   num_samples)`; return `(samples, info)` with samples shaped
   `(num_samples, dim)`. `NLEPosterior.sample` expands to
   `(num_samples, dim, 1)` exactly as for the other samplers.

## Error handling

- Outer loop hitting `max_iterations` → `RuntimeError` naming the likely
  fixes (looser `dlogz`, larger `num_live`), not a silent hang.
- Constructor validation: `num_live > 0`; resolved `num_delete` in
  `[1, num_live)`.
- blackjax imported lazily inside `run()`, matching the module convention.

## Testing — `tests/inference/test_nested.py`

CPU (`JAX_PLATFORMS=cpu`), mock flows mirroring `test_smc.py`
(`GaussianMock`, `BimodalMock`):

1. **Gaussian analytic recovery** — posterior mean ≈ `x_o / 2`; shape
   `(num_samples, dim)` via the `Sampler` contract and `(n, dim, 1)` via
   `NLEPosterior.sample`.
2. **Bimodal recovery** — both modes populated (fractions > 0.3 each).
3. **Evidence correctness** — the Gaussian mock has closed-form
   log-evidence: the prior (`make_gaussian_prior`, numpyro) is normalized
   but `GaussianMock.log_prob = −½‖x−θ‖²` omits the Gaussian constant, so
   `log Z = −‖x_o‖²/4 − (dim/2)·log 2`. Assert `log_evidence` within
   ~3·`log_evidence_err`. This is the capability the other samplers
   don't have.
4. **Info contract** — all fields present and finite; `dead` non-None;
   `ess > 0`; `num_dead > 0`.
5. **Defaults** — `num_inner_steps=None` resolves to `max(5, 2·dim)`;
   `num_delete=None` resolves to `num_live // 10`.
6. **Validation** — bad `num_live` / `num_delete` raise `ValueError`.
7. **Pipeline wiring** — `return_info=True` returns
   `(samples, NestedSamplerInfo)` through `NLEPosterior.sample`.

## Out of scope

- Anesthetic plots/integration (enabled later by the retained `dead` field).
- Recovery-script (`scripts/*_nle_recovery.py`) wiring.
- Docs page updates.
- Making NS the default sampler.
