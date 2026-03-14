# GenSBI Solver Refactoring — Status

## Current Status: ✅ 522/522 tests pass (Steps 3a–3c complete, Prior refactoring complete)

All pipeline dispatch uses the new solver classes. The old classes and their tests have been deleted. No test failures. All new solver files have 100% coverage from pipeline tests.

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
| `flow_matching.py` | `ODESolver` → `NewFMODESolver`, `BaseFmSDESolver` → `NewSDESolver`. All imports are top-level. |
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

### Step 3b-prior: Prior Refactoring

Replaced all custom prior classes with standard numpyro distributions. The prior lives on the method, `event_shape` is passed via `build_path`.

#### Core changes:
| File | Change |
|------|--------|
| `core/prior.py` | **[NEW]** `make_gaussian_prior` factory + `is_gaussian_prior` helper. Moved here from top-level `prior.py`. |
| `core/__init__.py` | Exports `make_gaussian_prior`, `is_gaussian_prior` |
| `core/generative_method.py` | `build_path(config, event_shape)`, `sample_init(key, nsamples)` (dropped `path` param). Added `has_custom_prior` property. |
| `core/flow_matching.py` | Deleted `StandardNormalPrior`. `__init__` accepts `prior=None`. `build_path` constructs/validates prior. `prepare_batch` uses `self.prior.sample()`. `build_solver` uses prior for SDE `mu0/sigma0` (with user kwargs override). `build_log_prob_fn` accepts optional `log_prior` override. |
| `core/score_matching.py` | Removed `VPPrior`/`VEPrior` usage. `build_path` auto-constructs `make_gaussian_prior` for VP/VE. `build_log_prob_fn` accepts optional `log_prior` override. |
| `core/diffusion_edm.py` | `build_path` constructs `make_gaussian_prior`. |
| `sm_prior.py` | `DeprecationWarning` on `VPPrior`/`VEPrior` (kept for now). |
| `prior.py` | **[DELETED]** — moved to `core/prior.py`. |

#### Pipeline changes:
| File | Change |
|------|--------|
| `conditional_pipeline.py` | `event_shape=(dim_obs, ch_obs)`, simplified `sample_init` call |
| `unconditional_pipeline.py` | Same as conditional |
| `joint_pipeline.py` | `event_shape=(dim_joint, ch_obs)` — prior on whole joint space. Sampler marginalizes by slicing `[:dim_obs]`. `get_log_prob_fn`/`log_prob` accept optional `prior` (numpyro dist): auto-constructs obs Gaussian if no custom prior, otherwise user must supply. |

#### Import cleanup:
- Removed all lazy imports in `flow_matching.py` (`NewFMODESolver`, `NewSDESolver`) — confirmed no circular dependency.
- Removed lazy import in `joint_pipeline.py` (`make_gaussian_prior`).
- All imports are now top-level.

### Step 3c: Delete Old Classes ✅

Deleted all deprecated old solver classes and their tests.

#### Source files deleted:
| File | Contents |
|------|----------|
| `flow_matching/solver/ode_solver.py` | Old `ODESolver` (365 lines) |
| `flow_matching/solver/sde_solver_fm.py` | Old `BaseFmSDESolver`, `ZeroEndsSolver`, `NonSingularSolver` (464 lines) |
| `diffusion/solver/sm_solver.py` | Old `SMSolver`, `SMPFSolver` (228 lines) |
| `diffusion/solver/sm_samplers.py` | Old `sm_reverse_sde_sampler` (245 lines) |

#### Test files deleted:
| File | Contents |
|------|----------|
| `test_solver_refactor_equivalence.py` | Old↔new equivalence tests (616 lines) |
| `test_ode_solver_flow_matching.py` | Old `ODESolver` tests (151 lines) |
| `test_sde_solver_flow_matching.py` | Old FM SDE solver tests (236 lines) |

#### Files modified:
| File | Change |
|------|--------|
| `flow_matching/solver/__init__.py` | Removed `ODESolver`, `BaseFmSDESolver`, `ZeroEndsSolver`, `NonSingularSolver` exports |
| `diffusion/solver/__init__.py` | Removed `SMSolver`, `SMPFSolver` exports |
| `test_sm_solver.py` | Removed old-class unit tests, kept prior tests and pipeline-level tests using new solvers |

---

## Key Design Decisions

### 1. Prior on the method, not the path
The prior is stored on `GenerativeMethod.prior`, constructed in `build_path`. The path is a pure mathematical interpolation primitive. This decouples prior logic from path construction.

### 2. Joint pipeline prior space
The joint prior lives on `(dim_joint, ch)` because obs/cond roles are a runtime choice (condition mask). For sampling, marginalize by slicing obs dims. For `log_prob`, auto-construct obs marginal if default Gaussian, otherwise require user to pass the marginal prior.

### 3. `has_custom_prior` flag
`GenerativeMethod.has_custom_prior` (property) checks whether `_user_prior` was set at construction. Used by joint pipeline to decide whether `log_prob` can auto-construct the obs marginal or must require an explicit prior.

### 4. SDE solver mu0/sigma0 priority
In `build_solver`, prior provides default `mu0`/`sigma0`; user kwargs override. This is necessary for the joint pipeline where the prior is on `(dim_joint, ch)` but the SDE solver operates in obs-space `(dim_obs, ch)`.

### 5. SM SDE solver eager build ✅ (resolved)
`NewSMSDESolver` no longer needs `mu0`/`sigma0`. Constructor is `(velocity_model, sde, eps0)`. `build_sampler_fn` now builds eagerly.

### 6. `NewSDESolver.get_sampler` shape inference
The `get_sampler` method infers `_flat_dim` and `_sample_shape` from `x_init` inside the sampler closure.

### 7. `flat_dim` from `y_flat.shape[0]`
All `get_diffusion()` closures compute `flat_dim = y_flat.shape[0]` inside the closure from the argument.

---

## What Remains

### Step 3c: Delete Old Classes ✅
- [x] Delete `ODESolver` from `ode_solver.py`
- [x] Delete `BaseFmSDESolver`, `ZeroEndsSolver`, `NonSingularSolver` from `sde_solver_fm.py`
- [x] Delete `SMSolver`, `SMPFSolver` from `sm_solver.py`
- [x] Delete `sm_samplers.py`
- [x] Delete `test_solver_refactor_equivalence.py` (equivalence tests become obsolete)
- [x] Delete old ODE solver tests: `test_ode_solver_flow_matching.py` (tests old `ODESolver`)
- [x] Delete old FM SDE solver tests: `test_sde_solver_flow_matching.py`
- [x] Update all `__init__.py` exports to remove old class names
- [x] Rewrite `test_sm_solver.py` (remove old-class tests, keep pipeline tests)
- [x] Run full test suite — 522 passed, 95% coverage
- [x] Verify new solver coverage — all 4 new solver files at 100%

### Step 3d: Remove "New" Prefix
- [ ] Rename `NewFMODESolver` → `FMODESolver`
- [ ] Rename `NewFMSDESolver` → `FMSDESolver`
- [ ] Rename `NewZeroEndsSolver` → `ZeroEndsSolver`
- [ ] Rename `NewNonSingularSolver` → `NonSingularSolver`
- [ ] Rename `NewSMODESolver` → `SMODESolver`
- [ ] Rename `NewSMSDESolver` → `SMSDESolver`
- [ ] Rename `NewSDESolver` → `SDESolver`
- [ ] Rename `NewODESolver` → `ODESolver`
- [ ] Update all imports in source and test files
- [ ] Run full test suite

### Future: Arbitrary Conditionals (joint pipeline)
The joint pipeline currently assumes positional obs/cond split: obs = first `dim_obs` dims, cond = remaining. Both the sampler marginalizer (`[:dim_obs]` slice) and `JointWrapper.conditioned()` share this assumption. For arbitrary condition masks, both must be updated together. Flagged with comments in the code.

### Future: Delete VPPrior/VEPrior
`sm_prior.py` classes have deprecation warnings. Can be deleted once no external users depend on them.
