# GenSBI Solver Refactoring — Step 3 Status

## Current Status: ✅ 589/589 tests pass (Step 3a–3b complete)

All pipeline dispatch now uses the new solver classes. The old classes still exist with deprecation warnings and are tested by the equivalence tests. No test failures.

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
| `score_matching.py` | `SMSolver`/`SMPFSolver` → `NewSMSDESolver`/`NewSMODESolver`. SM SDE solver built **lazily** inside sampler closure. |
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

---

## Key Design Decisions

### 1. Lazy imports in `flow_matching.py`
Circular import: `fm_ode_solver.py` → `core.ode_solver` → `core.__init__` → `flow_matching.py` → `fm_ode_solver.py`.
**Fix**: `NewFMODESolver` and `NewSDESolver` are imported inside methods, not at module level.

### 2. Lazy SM SDE solver build in `score_matching.py`
`NewSMSDESolver` requires `mu0`/`sigma0` at init (for `flat_dim` computation), but `build_solver` doesn't know sample dimensions.
**Fix**: `build_sampler_fn` builds the SDE solver inside the sampler closure where `x_init.shape` is known.

> **IMPORTANT**: This lazy build is a workaround. The pipeline knows `dim_obs` and `ch_obs` — a cleaner approach would be to pass those through as solver kwargs. This should be addressed when removing the "New" prefix (Step 3d).

### 3. `NewSDESolver.get_sampler` shape inference
The `get_sampler` method now infers `_flat_dim` and `_sample_shape` from `x_init` inside the sampler closure, rather than relying on pre-computed values from `mu0`. This makes the solver robust to placeholder `mu0`/`sigma0`.

---

## What Remains (Steps 3c–3d)

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
- [ ] Address the SM SDE solver `mu0`/`sigma0` design: pass `dim_obs`/`ch_obs` from pipeline
- [ ] Update all imports in source and test files
- [ ] Run full test suite
