# Joint Pipeline SDE Solver Debugging Handout

## Problem

6 tests fail when using SDE-based solvers with JointPipeline. The same solvers pass for ConditionalPipeline and UnconditionalPipeline.

### Affected Tests

| Test file | Test | Solver | Pipeline | Status |
|-----------|------|--------|----------|--------|
| `test_fm_solver.py` | `test_joint_fm_sde_solver` | `ZeroEndsSolver` | Joint | ❌ |
| `test_fm_solver.py` | `test_joint_fm_sde_solver` | `NonSingularSolver` | Joint | ❌ |
| `test_sm_solver_pipelines.py` | `test_joint_sm_default_solver` | `SMSolver` | Joint | ❌ |
| `test_sm_solver_pipelines.py` | `test_joint_sm_default_solver` | `SMSolver` | Joint | ❌ |
| `test_sm_solver_pipelines.py` | `test_joint_sm_pf_solver` | `SMPFSolver` | Joint | ❌ |
| `test_sm_solver_pipelines.py` | `test_joint_sm_pf_solver` | `SMPFSolver` | Joint | ❌ |

### Passing Tests (Same Solvers, Different Pipelines)

All equivalent tests pass for Conditional and Unconditional pipelines — 13 tests total:
- `test_unconditional_fm_sde_solver` × 2 solvers ✅
- `test_conditional_fm_sde_solver` × 2 solvers ✅
- `test_unconditional_sm_default_solver` × 2 SDE types ✅
- `test_conditional_sm_default_solver` × 2 SDE types ✅
- `test_unconditional_sm_pf_solver` × 2 SDE types ✅
- `test_conditional_sm_pf_solver` × 2 SDE types ✅
- `test_unconditional_fm_ode_custom_time_grid` ✅

## Exact Error

```
ValueError: Terms are not compatible with solver! Got:
MultiTerm(
  terms=(
    ODETerm(vector_field=<function ...drift_flat>),
    ControlTerm(
      vector_field=<function ...diff_flat>,
      control=VirtualBrownianTree(
        t0=f32[](jax), t1=f32[](jax), tol=f32[](jax),
        shape=ShapeDtypeStruct(shape=(4,), dtype=float32),
        levy_area=diffrax.BrownianIncrement,
        key=u32[2](jax), _spline='sqrt'
      )
    )
  )
)
but expected:
diffrax.AbstractTerm
```

The error comes from `diffrax._integrate.py:205` — diffrax checks that the `terms` structure matches the solver's `term_structure`. The SDE solvers produce `MultiTerm(ODETerm, ControlTerm)`, but the selected diffrax solver (e.g. `Euler`) expects a single `AbstractTerm`.

## Key Observation

> **The `shape=(4,)` in the error equals `dim_obs * ch_obs = 2 * 2 = 4`** which is correct for the joint pipeline's sample space. The shapes appear correct.

## Code Path Analysis

### `ConditionalPipeline.get_sampler` vs `JointPipeline.get_sampler`

**These are structurally identical.** Both:
1. Select `model_wrapped` (ema or regular)
2. Build `extras = {cond, obs_ids, cond_ids}`
3. Call `self.method.build_sampler_fn(model_wrapped, self.path, extras, **sampler_kwargs)`
4. Return a `sampler(key, nsamples)` closure

Source: [conditional_pipeline.py:L234-285](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py#L234-L285) and [joint_pipeline.py:L360-410](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py#L360-L410)

### FM `build_sampler_fn` (both pipelines use the same code)

At [flow_matching.py:L161-213](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/flow_matching.py#L161-L213):
1. `build_solver(model_wrapped, path, solver=solver)` — constructs the solver instance
2. `pass_key = isinstance(solver_instance, BaseFmSDESolver)` — True for SDE solvers
3. `solver_instance.get_sampler(method="Euler", ...)` — **always passes `method="Euler"`**
4. Returns `sampler_fn(key, x_init)`

### FM `build_solver` (where the solver is instantiated)

At [flow_matching.py:L114-140](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/flow_matching.py#L114-L140):
```python
if solver is None:
    solver = self.get_default_solver()
solver_cls, solver_kwargs = solver
return solver_cls(velocity_model=model_wrapped, **solver_kwargs)
```

For our FM SDE tests, `solver=(ZeroEndsSolver, {"mu0": ..., "sigma0": ..., "alpha": 1.0})`.

### `BaseFmSDESolver.get_sampler` (where the diffrax integration call lives)

At [sde_solver_fm.py:L125-298](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/flow_matching/solver/sde_solver_fm.py#L125-L298):
- Creates `MultiTerm(ODETerm(drift_flat), ControlTerm(diff_flat, brownian_motion))`
- Uses `diffrax.Euler()` as the default diffrax solver (from `method="Euler"`)
- Calls `diffeqsolve(terms, solver, ...)`

### SM `build_sampler_fn` (same pattern)

At [score_matching.py:L159-198](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/score_matching.py#L159-L198):
- Calls `self.build_solver(model_wrapped, path, solver=solver)`
- Then `solver_instance.get_sampler(nsteps=..., model_extras=...)`

## What I Tried

1. Ran all 19 solver tests — 13 pass (conditional + unconditional), 6 fail (all joint)
2. Compared `ConditionalPipeline.get_sampler` and `JointPipeline.get_sampler` — they are structurally identical
3. Verified the `solver_kwargs` shapes match what each pipeline expects
4. Got full traceback with `--runxfail` and `JAX_TRACEBACK_FILTERING=off`
5. Marked the 6 tests as `xfail(strict=True)` as temporary workaround

## What I Did NOT Investigate Yet

1. **Whether the conditional test is _actually_ hitting the SDE path**: The conditional test passes, but I haven't added print statements or breakpoints to confirm it truly goes through the `BaseFmSDESolver`/`MultiTerm` code path. It's possible that `build_solver` resolves differently for conditional vs joint due to how `solver` kwarg flows through the pipeline.

2. **The model wrapper behavior difference**: `ConditionalModelWrapper` and `JointModelWrapper` may produce outputs with different shapes or behaviors that cause the traced diffrax solver check to diverge. This is worth investigating because the error is at trace-time (shape checking), not runtime.

3. **Whether `diffrax.Euler` with `MultiTerm` actually works at all**: Looking at the code, `Euler` has `term_structure = AbstractTerm`, which means it should NOT support `MultiTerm`. If this is true, then the conditional/unconditional tests passing is the anomaly, not the joint tests failing.

4. **Whether there is a JIT compilation difference**: Joint pipeline may be compiling the sampler differently (e.g., due to different static arguments or traced values like `t0`, `t1`).

## Suggested Debugging Plan

### Step 1: Confirm conditional actually uses SDE solver
Add print/assert inside `BaseFmSDESolver.get_sampler.sample_one` to verify the conditional test truly executes this code path. If conditional doesn't hit this path, the bug is in how `build_solver` routes things differently.

### Step 2: Minimal reproduction
Write a minimal script (no pipeline, no test framework) that:
1. Creates a `MockConditionalModel` wrapped in `ModelWrapper`
2. Instantiates `ZeroEndsSolver(velocity_model=..., mu0=..., sigma0=..., alpha=...)`
3. Calls `solver.get_sampler(method="Euler", ...)`
4. Calls the returned sampler

Then do the same with `MockJointModel` wrapped in `JointModelWrapper`. See which step diverges.

### Step 3: Check `diffrax.Euler` compatibility
Verify in diffrax source whether `Euler.term_structure` supports `MultiTerm`. If it doesn't, investigate why conditional/unconditional tests pass — they may use a different code path (e.g., `ODESolver` instead of `BaseFmSDESolver`).

### Step 4: Check wrapper output shapes
Compare the output shapes of `ConditionalModelWrapper.__call__` vs `JointModelWrapper.__call__` during tracing — different shapes could cause diffrax's `filter_eval_shape` to take different branches.

## Key Files

| Purpose | Path |
|---------|------|
| FM SDE solver test file | [test_fm_solver.py](file:///data/users/Aurelio/Github/GenSBI/tests/recipes/test_fm_solver.py) |
| SM solver test file | [test_sm_solver_pipelines.py](file:///data/users/Aurelio/Github/GenSBI/tests/recipes/test_sm_solver_pipelines.py) |
| FM `build_sampler_fn` / `build_solver` | [flow_matching.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/flow_matching.py) |
| SM `build_sampler_fn` / `build_solver` | [score_matching.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/score_matching.py) |
| FM SDE solver (where MultiTerm is built) | [sde_solver_fm.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/flow_matching/solver/sde_solver_fm.py) |
| Conditional pipeline sampler | [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py) |
| Joint pipeline sampler | [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py) |
| Mock models | [mock_models.py](file:///data/users/Aurelio/Github/GenSBI/tests/recipes/mock_models.py) |
