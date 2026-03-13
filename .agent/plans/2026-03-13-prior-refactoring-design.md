# Prior Refactoring: Single Source of Truth via numpyro Distributions

## Problem

The prior (the distribution from which initial noise `x_init` is sampled) is currently scattered across multiple locations with no enforced consistency:

- **FM**: `StandardNormalPrior` on `FlowMatchingMethod` (shape-agnostic, always N(0,I))
- **SM**: `VPPrior`/`VEPrior` on `ScoreMatchingMethod` (shape-agnostic, determined by SDE type)
- **FM SDE solvers**: `mu0`/`sigma0` passed as constructor kwargs (used in `get_score()` formula)
- **Core SDE solver**: `mu0`/`sigma0` used only for `flat_dim`/`sample_shape`; `prior_distribution` attribute is never read

This means:
1. FM SDE solver's `mu0`/`sigma0` and the pipeline's `StandardNormalPrior` are disconnected — they agree only by convention
2. SM SDE solver requires `mu0`/`sigma0` at init but doesn't use them mathematically, forcing a lazy-build workaround
3. Users must manually keep solver kwargs (`mu0`/`sigma0`) consistent with the training prior

## Design

### Core Idea

The prior becomes a **standard numpyro distribution** living on the **path**, constructed with correct dimensions by the **pipeline**. The prior is the single source of truth for:
- Sampling `x_init` (via `prior.sample(key, (nsamples,))`)
- Computing `log_prob` (via `prior.log_prob(x)`)
- Extracting `mu0`/`sigma0` for FM SDE solver math (via `prior.base_dist.loc`/`.scale`)

### Prior Construction

A convenience factory creates the numpyro distribution:

```python
# gensbi/prior.py (new file)
import jax.numpy as jnp
import numpyro.distributions as dist

def make_gaussian_prior(dim, ch, mu=0.0, sigma=1.0):
    """Create a Gaussian prior as a numpyro distribution.
    
    Returns Independent(Normal(loc, scale), 2) with event_shape=(dim, ch).
    """
    loc = jnp.full((dim, ch), mu)
    scale = jnp.full((dim, ch), sigma)
    return dist.Independent(dist.Normal(loc, scale), 2)
```

Users can pass **any** `numpyro.distributions.Distribution` as a prior. The factory is just a convenience for the common Gaussian case.

### Where the Prior Lives

```
Pipeline.__init__
  ├── knows dim_obs, ch_obs
  ├── constructs prior = make_gaussian_prior(dim_obs, ch_obs)  [default]
  │   (or user passes a custom numpyro distribution)
  ├── method.set_prior(prior) or passes prior to build_path
  └── self.path = method.build_path(config)  ← path gets .prior attribute
```

Specifically:
- **`GenerativeMethod.build_path(config, prior=None)`** gains an optional `prior` parameter
- The pipeline passes the prior: `self.path = method.build_path(config, prior=prior)`
- The path stores it: `self.path.prior = prior`
- For SM, if no prior is passed, it's derived from the SDE type and scheduler config

### How Each Component Uses the Prior

| Component | Before | After |
|-----------|--------|-------|
| **Pipeline `get_sampler`** | `self.method.sample_init(key, (n, dim, ch), self.path)` | `self.path.prior.sample(key, (n,))` — no more passing explicit dims |
| **FM `prepare_batch`** | `self.prior.sample(key, x_1.shape)` | `self.path.prior.sample(key, (batch,))` — or keep shape-agnostic with a helper |
| **FM `build_solver`** (ODE) | `solver_cls(velocity_model=wrapped, **kwargs)` | No change |
| **FM `build_solver`** (SDE) | User passes `mu0`/`sigma0` as kwargs | `build_solver` extracts from `path.prior.base_dist.loc`/`.scale` |
| **SM `build_solver`** (SDE) | Lazy build with placeholder mu0/sigma0 | `solver_cls(velocity_model=wrapped, sde=sde)` — no mu0/sigma0 needed |
| **SM `build_solver`** (ODE) | Works as-is | No change |
| **`build_log_prob_fn`** | `self.prior.log_prob` | `self.path.prior.log_prob` |

### Solver Changes

**`NewSDESolver`**: **Remove** `mu0`/`sigma0` entirely from the constructor. Remove `self.prior_distribution`, `self.sample_shape`, and `self.flat_dim` — none of these are needed. The base solver becomes a pure integrator. Constructor signature: `NewSDESolver(velocity_model, eps0=1e-5)`.

- `get_sampler()` already infers shapes from `x_init` at runtime (via `_flat_dim = math.prod(x_init.shape[1:])`).
- `get_diffusion()` subclass implementations infer `flat_dim` from `y_flat.shape[0]` at runtime — `y_flat` is the 1D flattened state inside the vmapped `sample_one` and its shape is statically known under JAX tracing.

**`NewFMSDESolver`**: **Owns** `mu0`/`sigma0` — moved down from the base class. Only needed for the `get_score()` formula: `score = (-t·vf + mu0 - x) / ((1-t)·sigma0²)`. Gets them from `path.prior` via `build_solver`, not from user kwargs. `get_diffusion()` uses `y_flat.shape[0]` instead of `self.flat_dim`. Constructor signature: `NewFMSDESolver(velocity_model, mu0, sigma0, eps0=1e-5)`.

**`NewSMSDESolver`**: No `mu0`/`sigma0` at all. `get_diffusion()` uses `y_flat.shape[0]`. Constructor signature: `NewSMSDESolver(velocity_model, sde, eps0=1e-3)`. This **eliminates the lazy-build workaround** in `build_sampler_fn` — the solver can now be constructed eagerly since it no longer needs shape information at init.

Class hierarchy:
```
NewSDESolver(velocity_model, eps0)              ← pure integrator, no shape attrs
  ├── NewFMSDESolver(vm, mu0, sigma0, eps0)     ← owns mu0/sigma0 for get_score()
  │     ├── NewZeroEndsSolver(... + alpha)
  │     └── NewNonSingularSolver(... + alpha)
  └── NewSMSDESolver(vm, sde, eps0)             ← no mu0/sigma0
```

> [!NOTE]
> `flat_dim` is only needed by SDE solvers (not ODE solvers) because the diffusion function must return a `(flat_dim, flat_dim)` matrix for `diffrax.ControlTerm`. ODE solvers have no Brownian motion and no diffusion matrix.

### Solver Compatibility Validation

When the user passes a different solver at sampling time (e.g., `ZeroEndsSolver` for a model trained with FM ODE), `build_solver` or `build_sampler_fn` validates:

- **FM SDE solvers** require a Gaussian prior → check `isinstance(prior.base_dist, dist.Normal)` (or similar)
- **SM SDE solver** works with any prior that matches the SDE type
- **ODE solvers** work with any prior

This validation catches mismatches early with a clear error message.

### `prepare_batch` Handling

FM's `prepare_batch` currently uses `self.prior.sample(key, x_1.shape)` where shape is `(batch, dim, ch)`. With a numpyro distribution, the event shape `(dim, ch)` is baked in, so we need `self.prior.sample(key, (batch,))`.

**Option**: Keep `prepare_batch` using the prior from the method/path. Since `x_1.shape[0]` gives the batch size, this is a clean 1-line change:
```python
x_0 = path.prior.sample(rng_x0, (x_1.shape[0],))  # or self.path.prior
```

For SM, `prepare_batch` currently uses `jax.random.normal(key, x_1.shape)` directly (always N(0,I) noise for the forward process). This is conceptually correct — the training noise is always standard normal regardless of the prior — so it should remain as-is.

## Files Changed

### New
- `src/gensbi/prior.py` — `make_gaussian_prior()` factory, helper to check if a prior is Gaussian

### Modified (Source)
| File | Changes |
|------|---------|
| `core/generative_method.py` | `build_path` signature gains `prior=None`; `sample_init` simplified or removed |
| `core/flow_matching.py` | Remove `StandardNormalPrior` class. `build_path` accepts prior, attaches to path. `prepare_batch` uses `path.prior`. `build_solver` extracts mu0/sigma0 from `path.prior` for SDE solvers. `build_log_prob_fn` uses `path.prior.log_prob`. |
| `core/score_matching.py` | Remove `VPPrior`/`VEPrior` usage from `__init__`. `build_path` constructs prior from SDE config. `build_solver` (SDE case) stops passing mu0/sigma0. `build_sampler_fn` builds SDE solver **eagerly** (lazy closure eliminated — solver no longer needs shape at init). `build_log_prob_fn` uses `path.prior.log_prob`. |
| `core/sde_solver.py` | **Remove** `mu0`/`sigma0` from constructor entirely. Remove `self.prior_distribution`, `self.sample_shape`, `self.flat_dim`. `get_sampler` infers all shapes from `x_init`. |
| `diffusion/solver/sm_sde_solver_new.py` | Drop `mu0`/`sigma0` from constructor — only takes `velocity_model`, `sde`. `get_diffusion()` infers `flat_dim` from `y_flat.shape[0]`. |
| `flow_matching/solver/fm_sde_solver.py` | `get_score()` reads `self.mu0`/`self.sigma0` (no change — still passed by `build_solver`). `get_diffusion()` in `NewZeroEndsSolver`/`NewNonSingularSolver` changed to use `y_flat.shape[0]` instead of `self.flat_dim`. |
| `flow_matching/path/affine_prob_path.py` | Add optional `.prior` attribute. |
| `diffusion/path/sm_path.py` | Add `.prior` attribute (derived from scheduler). |
| `recipes/conditional_pipeline.py` | Construct prior in `__init__`, pass to `build_path`. Simplify `get_sampler` to use `path.prior.sample`. |
| `recipes/joint_pipeline.py` | Same as conditional. |
| `recipes/unconditional_pipeline.py` | Same as conditional. |

### Deleted
- `diffusion/sm_prior.py` — `VPPrior`/`VEPrior` replaced by numpyro distributions via `make_gaussian_prior`

### Modified (Tests)
| File | Changes |
|------|---------|
| `test_solver_fm_pipelines.py` | Remove `_sde_solver_kwargs()` — mu0/sigma0 no longer in solver kwargs. Only `alpha` remains. |
| `test_solver_sm_pipelines.py` | Minimal changes — SM tests don't pass mu0/sigma0 already. |
| `test_generative_method.py` | Update mocks/assertions for new `build_path` signature. |
| `test_sde_solver.py` | Update solver construction (mu0/sigma0 optional). |
| `test_sm_solver.py` | Update solver construction. |
| `test_sde_solver_flow_matching.py` | Update solver construction. |

## Verification Plan

### Automated Tests

Run the full existing test suite to verify no regressions:

```bash
cd /data/users/Aurelio/Github/GenSBI && uv run python -m pytest tests/ -x -q
```

Key test files to focus on:
- `tests/recipes/test_solver_fm_pipelines.py` — FM ODE + SDE across all 3 pipeline types
- `tests/recipes/test_solver_sm_pipelines.py` — SM SDE + ODE across all 3 pipeline types
- `tests/diffusion/solver/test_sde_solver.py` — core SDE solver
- `tests/diffusion/solver/test_sm_solver.py` — SM solver
- `tests/flow_matching/solver/test_sde_solver_flow_matching.py` — FM SDE solver
- `tests/core/test_generative_method.py` — method strategy tests

### New Tests to Add
1. Test that `make_gaussian_prior` produces correct distribution (sample shape, log_prob)
2. Test that FM SDE solver raises if prior is non-Gaussian
3. Test that SM SDE solver works without mu0/sigma0
4. Test that `path.prior` is correctly propagated from pipeline

## Scope Considerations

> [!IMPORTANT]
> This refactoring is **orthogonal** to the ongoing solver rename (Steps 3c–3d in the handout). It can be done before or after the rename. Doing it **before** the rename means we change the `New*` classes; doing it **after** means we change the final-name classes. Either order works.

> [!NOTE]
> EDM (`DiffusionEDMMethod`) uses `path.sample_prior(key, shape)` which delegates to the scheduler. EDM is unaffected by this refactoring — its prior is handled correctly already. We should not change EDM as part of this work.
