# GenSBI Solver Refactoring — Status

## Current Status: ✅ 594/594 tests pass (Steps 3a–3b complete, Prior refactoring complete)

All pipeline dispatch uses the new solver classes. The old classes still exist with deprecation warnings and are tested by the equivalence tests. No test failures.

---

## What Was Done

### Step 3a: Deprecation Warnings
Added `DeprecationWarning` to all old solver `__init__` methods:
- `ODESolver` → "use FMODESolver"
- `BaseFmSDESolver` (covers `ZeroEndsSolver`, `NonSingularSolver`) → "use the new solver classes"
- `SMSolver` → "use SMSDESolver"
- `SMPFSolver` → "use SMODESolver"

### Step 3b: Pipeline Replacement

#### Source files modified:
| File | Change |
|------|--------|
| `flow_matching.py` | `ODESolver` → `NewFMODESolver`, `BaseFmSDESolver` → `NewSDESolver`. **Lazy imports** to break circular dependency. |
| `score_matching.py` | `SMSolver`/`SMPFSolver` → `NewSMSDESolver`/`NewSMODESolver`. SM SDE solver now built **eagerly** (no lazy workaround needed). |
| `unconditional_pipeline.py` | Removed unused old solver re-exports |
| `conditional_pipeline.py` | Removed unused old solver re-exports |
| `joint_pipeline.py` | Removed unused old solver re-exports |
| `sde_solver.py` | `get_sampler` infers `_flat_dim`/`_sample_shape` from `x_init` |

#### Test files modified:
| File | Change |
|------|--------|
| `test_generative_method.py` | Updated assertions and mocks to new class names |
| `test_solver_fm_pipelines.py` | `ZeroEndsSolver` → `NewZeroEndsSolver`, `NonSingularSolver` → `NewNonSingularSolver` |
| `test_solver_sm_pipelines.py` | `SMSolver` → `NewSMSDESolver`, `SMPFSolver` → `NewSMODESolver` |
| `test_sm_solver.py` | Updated pipeline-level test functions to use new classes |

### Step 3b-prior: Prior Refactoring (solver-side)

Solved the `mu0`/`sigma0` design issue that was blocking Step 3d. The base SDE solver is now a pure integrator with no shape-related attributes.

#### What was the problem?
`NewSDESolver.__init__` required `mu0`/`sigma0` (prior mean/std) to compute `flat_dim`, `sample_shape`, and `prior_distribution`. This forced the SM SDE solver to use a **lazy build** workaround (constructing the solver inside the sampler closure where `x_init.shape` was known).

#### What was done:
| File | Change |
|------|--------|
| `prior.py` | **[NEW]** `make_gaussian_prior` factory + `is_gaussian_prior` helper |
| `tests/test_prior.py` | **[NEW]** 5 tests for the prior factory |
| `sde_solver.py` | Stripped `mu0`/`sigma0`/`flat_dim`/`sample_shape`/`prior_distribution` from `NewSDESolver`. Constructor is now `(velocity_model, eps0)`. |
| `fm_sde_solver.py` | Added `__init__` to `NewFMSDESolver` — it owns `mu0`/`sigma0` (needed for score derivation). All `get_diffusion()` closures use `y_flat.shape[0]` instead of `self.flat_dim`. |
| `sm_sde_solver_new.py` | Dropped `mu0`/`sigma0` from constructor. `get_diffusion()` uses `y_flat.shape[0]`. |
| `score_matching.py` | Simplified `build_solver` SDE path (no placeholder mu0/sigma0). Converted `build_sampler_fn` SDE path from **lazy to eager** build. |
| `test_solver_refactor_equivalence.py` | Removed `mu0`/`sigma0` from `NewSMSDESolver` test construction. |

---

## Key Design Decisions

### 1. Lazy imports in `flow_matching.py`
Circular import: `fm_ode_solver.py` → `core.ode_solver` → `core.__init__` → `flow_matching.py` → `fm_ode_solver.py`.
**Fix**: `NewFMODESolver` and `NewSDESolver` are imported inside methods, not at module level.

### 2. SM SDE solver eager build ✅ (resolved)
~~`NewSMSDESolver` requires `mu0`/`sigma0` at init, but `build_solver` doesn't know sample dimensions.~~
**Resolved:** `NewSMSDESolver` no longer needs `mu0`/`sigma0`. Constructor is `(velocity_model, sde, eps0)`. `build_sampler_fn` now builds eagerly.

### 3. `NewSDESolver.get_sampler` shape inference
The `get_sampler` method infers `_flat_dim` and `_sample_shape` from `x_init` inside the sampler closure. The base solver stores no shape-related attributes at all.

### 4. `flat_dim` from `y_flat.shape[0]`
All `get_diffusion()` closures now compute `flat_dim = y_flat.shape[0]` inside the closure from the argument, eliminating the need for the solver to know the data shape at construction time.

---

## What Remains

### Step 3b-path: Prior-on-Path Migration (deferred)
Move the prior from the method strategy onto the path object, so the path is the single source of truth.

**Plan:** `.agent/plans/2026-03-13-prior-on-path-migration.md`

- [ ] Add `.prior` attribute to `AffineProbPath` (FM)
- [ ] Add `.prior` attribute to `SMPath` (SM, auto-constructs VP/VE)
- [ ] Update `FlowMatchingMethod` to use `path.prior` instead of `self.prior`
- [ ] Update `ScoreMatchingMethod` to use `path.prior` instead of `self.prior`
- [ ] Delete `sm_prior.py` (`VPPrior`/`VEPrior`) and `StandardNormalPrior`
- [ ] Update tests

### Step 3c: Delete Old Classes
- [ ] Delete `ODESolver` from `ode_solver.py`
- [ ] Delete `BaseFmSDESolver`, `ZeroEndsSolver`, `NonSingularSolver` from `sde_solver_fm.py`
- [ ] Delete `SMSolver`, `SMPFSolver` from `sm_solver.py`
- [ ] Delete `sm_samplers.py`
- [ ] Delete `test_solver_refactor_equivalence.py` (equivalence tests become obsolete)
- [ ] Delete old ODE solver tests: `test_ode_solver_flow_matching.py` (tests old `ODESolver`)
- [ ] Update all `__init__.py` exports to remove old class names
- [ ] Run full test suite

### Step 3d: Remove "New" Prefix
- [ ] Rename `NewFMODESolver` → `FMODESolver`
- [ ] Rename `NewFMSDESolver` → `FMSDESolver`
- [ ] Rename `NewZeroEndsSolver` → `ZeroEndsSolver`
- [ ] Rename `NewNonSingularSolver` → `NonSingularSolver`
- [ ] Rename `NewSMODESolver` → `SMODESolver`
- [ ] Rename `NewSMSDESolver` → `SMSDESolver`
- [ ] Rename `NewSDESolver` → `SDESolver`
- [ ] Rename `NewODESolver` → `ODESolver`
- [ ] Optionally add aliases in `__init__.py`
- [ ] Update all imports in source and test files
- [ ] Run full test suite
