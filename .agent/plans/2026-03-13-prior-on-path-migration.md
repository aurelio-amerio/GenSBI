# Prior-on-Path Migration — Implementation Plan

**Goal:** Move the prior distribution from the method strategy (`self.prior`) onto the path object (`path.prior`), making the path the single source of truth for the prior.

## Current State

| Framework | Prior location | `.sample` source | `.log_prob` source |
|-----------|---------------|-------------------|--------------------|
| FM | `FlowMatchingMethod.prior` (`StandardNormalPrior`) | `self.prior.sample` | `self.prior.log_prob` |
| SM | `ScoreMatchingMethod.prior` (`VPPrior`/`VEPrior`) | `path.sample_prior` → scheduler | `self.prior.log_prob` |

**Problem:** Prior lives on the method (not the path), and SM has the prior split across two locations — sampling goes through the scheduler while log_prob goes through the method's prior object.

## Proposed Changes

### Task 1: Add `.prior` to FM path (`AffineProbPath`)

#### [MODIFY] `src/gensbi/flow_matching/path/affine.py`

- Add a `prior` parameter to `AffineProbPath.__init__` with default `StandardNormalPrior()`.
- Store as `self.prior`.
- `CondOTProbPath` inherits this (its `__init__` calls `super` which sets the scheduler, but it doesn't pass a prior — default is fine since CondOT always uses N(0,I)).

---

### Task 2: Add `.prior` to SM path (`SMPath`)

#### [MODIFY] `src/gensbi/diffusion/path/sm_path.py`

- Add a `prior` parameter to `SMPath.__init__` with default `None`.
- If `prior is None`, auto-construct:
  - VP → `VPPrior()`
  - VE → `VEPrior(sigma_max=scheduler.sigma_max)`
- Store as `self.prior`.
- Keep existing `sample_prior()` method but redirect it to `self.prior.sample()` for consistency.

---

### Task 3: Update `FlowMatchingMethod` to use `path.prior`

#### [MODIFY] `src/gensbi/core/flow_matching.py`

- `prepare_batch`: change `self.prior.sample(...)` → `path.prior.sample(...)`
- `sample_init`: change `self.prior.sample(...)` → `path.prior.sample(...)`
- `build_log_prob_fn`: change `self.prior.log_prob` → `path.prior.log_prob`
- Remove `self.prior` from `__init__` and the `StandardNormalPrior` class (or keep the class if still imported elsewhere — check first)

---

### Task 4: Update `ScoreMatchingMethod` to use `path.prior`

#### [MODIFY] `src/gensbi/core/score_matching.py`

- `sample_init`: keep using `path.sample_prior(...)` (which now delegates to `path.prior.sample(...)`)
- `build_log_prob_fn`: change `self.prior.log_prob` → `path.prior.log_prob`
- Remove `self.prior` from `__init__` and the `VPPrior`/`VEPrior` construction in `build_path`
- Remove imports of `VPPrior`, `VEPrior` from `sm_prior.py`

---

### Task 5: Delete deprecated prior code

#### [DELETE] `src/gensbi/diffusion/sm_prior.py`

- Delete `VPPrior` and `VEPrior` classes (their logic now lives in `SMPath.__init__`)
- Or if still imported by tests/old code: add deprecation warnings first, delete in a follow-up

#### [MODIFY] `src/gensbi/core/flow_matching.py`

- If `StandardNormalPrior` is no longer needed as a standalone class on the method, remove it or move it to `prior.py` as a utility

> **IMPORTANT:** Check whether `StandardNormalPrior` or `VPPrior`/`VEPrior` are imported anywhere outside the method strategies before deleting. Use `grep_search` to verify.

---

### Task 6: Update tests

- Update any test that constructs `ScoreMatchingMethod` and checks `method.prior`
- Update any test that constructs `VPPrior`/`VEPrior` directly (equivalence tests reference them)
- Add new tests:
  - `AffineProbPath` has a `.prior` attribute with correct `.sample`/`.log_prob`
  - `SMPath` auto-constructs the correct prior (VP vs VE)
  - `SMPath.sample_prior()` still works (backward compat)

---

## Verification Plan

### Automated Tests

```bash
uv run pytest tests/ -x --tb=short
```

All existing tests must pass. Key test files:
- `tests/recipes/test_solver_fm_pipelines.py`
- `tests/recipes/test_solver_sm_pipelines.py`
- `tests/test_solver_refactor_equivalence.py`
- `tests/test_prior.py`
