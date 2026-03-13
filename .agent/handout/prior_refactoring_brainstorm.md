# Prior Refactoring Brainstorm — Status & Open Questions

## Status: Design finalized, ready for implementation

Full design doc: `.agent/plans/2026-03-13-prior-refactoring-design.md`

---

## What We Decided

1. **Prior = numpyro distribution, single source of truth, lives on the path**
   - `make_gaussian_prior(dim, ch, mu=0, sigma=1)` factory → `Independent(Normal(...), 2)`
   - Users can pass any `numpyro.distributions.Distribution` as a custom prior
   - Pipeline constructs the prior (it knows `dim_obs`/`ch_obs`) and passes it to `build_path`

2. **`NewSDESolver` base class drops `mu0`/`sigma0` AND `flat_dim`/`sample_shape`/`prior_distribution` entirely**
   - Becomes a pure integrator: `NewSDESolver(velocity_model, eps0)`
   - `mu0`/`sigma0` move down to `NewFMSDESolver` (needed for `get_score()` formula)
   - `NewSMSDESolver` never had a mathematical need for them — just takes `(velocity_model, sde, eps0)`
   - `flat_dim` is not an attribute of any solver — inferred from `y_flat.shape[0]` at runtime

3. **Prior is extracted automatically for FM SDE solvers**
   - `build_solver` reads `path.prior.base_dist.loc`/`.scale` to get `mu0`/`sigma0`
   - Users no longer pass `mu0`/`sigma0` as solver kwargs — only `alpha`

4. **SM SDE solver lazy-build workaround eliminated**
   - `NewSMSDESolver` no longer needs shape info at init → can be built eagerly in `build_sampler_fn`

---

## ✅ Resolved: `get_diffusion()` and `flat_dim`

### Original Concern

`get_diffusion()` closures captured `self.flat_dim` at build time. If `self.flat_dim` were wrong, the `jnp.eye(flat_dim)` matrix would be the wrong size.

### Analysis

This was **not a bug today** — both FM and SM paths guaranteed correct `flat_dim`. But it was a design smell: `flat_dim` is just `features * channels`, a property of the data shape, not something a solver should own.

### Resolution

**Approach A (adopted)**: Infer `flat_dim` from `y_flat.shape[0]` at runtime inside `get_diffusion()` closures:

```python
def g_tilde(t, y_flat, args):
    flat_dim = y_flat.shape[0]   # statically known under JAX tracing
    return g * jnp.eye(flat_dim)
```

**Why this works**: Inside the vmapped `sample_one`, `y_flat` is a 1D vector `(F*C,)` — the batch dimension was stripped by `vmap`. `y_flat.shape[0]` is a static integer during JAX tracing (since `x_init.shape` is known at `@jit` call time), so `jnp.eye(y_flat.shape[0])` compiles identically to `jnp.eye(42)`.

**Why `flat_dim` is only an SDE concern**: ODE solvers don't have a Brownian motion or diffusion matrix — they just have `ODETerm` where drift input/output shapes match naturally. Only SDE solvers need `flat_dim` for the `(flat_dim, flat_dim)` diffusion matrix required by `diffrax.ControlTerm`.

---

## Other Open Questions

1. **`prepare_batch` for FM**: Currently uses `self.prior.sample(key, x_1.shape)`. With numpyro dist, becomes `path.prior.sample(key, (x_1.shape[0],))`. But `prepare_batch` receives `path` as an argument — does it need to access `path.prior` or should `self.prior` on the method still exist as a reference?

2. **SM `prepare_batch`**: Uses `jax.random.normal(key, x_1.shape)` directly. SM training noise is always standard normal regardless of VE/VP prior. Should this also use `path.prior.sample()` for consistency, or keep it as-is since it's conceptually different (training noise ≠ prior)?

3. **Solver compatibility validation**: Exact mechanism for checking that an SDE solver at sampling time is compatible with the training prior. Type checking? Attribute checking? Where does the validation live?

4. **Ordering vs. ongoing solver rename**: This refactoring is orthogonal to Steps 3c–3d (delete old classes, rename `New*` → final names). Which should go first? Doing prior refactoring first means changing `New*` classes; doing rename first means cleaner names but two large refactors in sequence.

---

## Files to Reference

| File | Why |
|------|-----|
| `src/gensbi/core/sde_solver.py` | Base `NewSDESolver` — `get_sampler()`, `flat_dim`/`sample_shape` usage |
| `src/gensbi/core/score_matching.py` | SM pipeline — lazy SDE build, `build_sampler_fn`, `build_solver` |
| `src/gensbi/core/flow_matching.py` | FM pipeline — `StandardNormalPrior`, `build_solver`, `prepare_batch` |
| `src/gensbi/flow_matching/solver/fm_sde_solver.py` | FM SDE — `get_score()` uses `self.mu0`/`self.sigma0` |
| `src/gensbi/diffusion/solver/sm_sde_solver_new.py` | SM SDE — `get_diffusion()` captures `self.flat_dim` |
| `src/gensbi/diffusion/sm_prior.py` | `VPPrior`/`VEPrior` — to be replaced by numpyro dists |
| `tests/recipes/test_solver_fm_pipelines.py` | FM tests — `_sde_solver_kwargs()` pattern to remove |
| `tests/recipes/test_solver_sm_pipelines.py` | SM tests — minimal changes expected |
