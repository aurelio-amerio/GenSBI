# Handout: Dynamic `model_extras` via diffrax `args` — Current State & Remaining Work

## Goal

The sampler should **not need to know the value** (or even existence) of `model_extras` until the exact moment it calls the model. All runtime data (`cond`, `obs_ids`, `cond_ids`, `edge_mask`, ...) should be passed through diffrax's `args: PyTree[Any]` parameter to `diffeqsolve`, which forwards it to `vector_field(t, y, args)`. This enables compiling the sampler once and reusing it for different conditions without recompilation.

**Reference**: [diffrax `args` docs](https://docs.kidger.site/diffrax/api/diffeqsolve/#diffrax.diffeqsolve), [forcing example](https://docs.kidger.site/diffrax/examples/forcing/)

## What Was Done

### Files modified (16 total)

| File | What changed | Status |
|------|-------------|--------|
| [sde_solver_fm.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/flow_matching/solver/sde_solver_fm.py) | `diffeqsolve(args=model_extras)`, sampler accepts `model_extras` param | ⚠️ Partial |
| [sm_samplers.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/diffusion/solver/sm_samplers.py) | drift reads `**args` not closure `**model_kwargs`, `diffeqsolve(args=model_kwargs)` | ✅ Dynamic |
| [sm_solver.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/diffusion/solver/sm_solver.py) | `sample()` accepts `model_extras` param | ✅ Dynamic |
| [flow_matching.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/flow_matching.py) | `sampler_fn` threads `model_extras` to SDE solver | ⚠️ Partial |
| [score_matching.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/score_matching.py) | `sampler_fn` threads `model_extras` to SM solver | ✅ Dynamic |
| [diffusion_edm.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/diffusion_edm.py) | `sampler_fn` accepts but ignores `model_extras` | ❌ Not dynamic |
| [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py) | Sampler accepts `model_extras` override, warning for batch cond | ✅ Done |
| [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py) | Same | ✅ Done |
| [pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/pipeline.py) | `sample_batched` builds sampler once, loops with per-cond extras | ✅ Done |
| [ode_solver.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/flow_matching/solver/ode_solver.py) | **Not touched** — still bakes everything into closures | ❌ Not dynamic |
| [model_wrapping.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/utils/model_wrapping.py) | **Not touched** — already supports merging `args` + `kwargs` | ✅ Ready |
| Tests (5 files) | x_o batch sizes fixed, xfail removed, `sample_batched` tests added | ✅ Done |

## The Core Problem Still Remaining

`model_extras` is still passed to `build_sampler_fn()` → `solver.get_sampler()` at **sampler creation time**. Even though we added `args=model_extras` to `diffeqsolve`, the extras are also baked into closures through other paths:

### FM SDE path (`sde_solver_fm.py`)

```
build_sampler_fn(model_extras={"cond": ..., "obs_ids": ...})
  └─> solver.get_sampler(model_extras=extras)          # extras passed at creation
        ├─> self.get_f_tilde()                          # ✅ no extras baked (good)
        ├─> diffeqsolve(args=model_extras)              # ✅ dynamic args (good)
        └─> sampler(x_init, key, model_extras=extras)   # ✅ runtime param (good)
              └─ BUT: model_extras used as DEFAULT VALUE # ⚠️ initial value still baked in
```

**Issue**: `model_extras` is passed to `get_sampler()` to serve as the default value. This is mostly fine for the default path, but ideally `get_sampler()` shouldn't need it at all.

### FM ODE path (`ode_solver.py`)

```
build_sampler_fn(model_extras={"cond": ..., "obs_ids": ...})
  └─> solver.get_sampler(model_extras=extras)          # extras passed at creation
        └─> ODETerm(self.velocity_model.get_vector_field(**model_extras))  # ❌ BAKED
            └─> vf closure captures kwargs=model_extras permanently
                diffeqsolve(args=None)                  # ❌ args not used
```

**Issue**: The ODE solver bakes `model_extras` into `get_vector_field(**model_extras)` as closure kwargs. `diffeqsolve(args=None)` — `args` is not used at all. This path cannot handle dynamic extras.

### EDM path (`diffusion_edm.py` → `edm_solver.py`)

```
build_sampler_fn(model_extras={"cond": ..., "obs_ids": ...})
  └─> solver.get_sampler(model_extras=extras)          # extras passed at creation
        └─> sampler_(key, x_init)                      # ❌ no model_extras param
                                                        # EDM solver uses discrete steps,
                                                        # not diffeqsolve, so no `args`
```

**Issue**: EDM solver has its own discrete sampling loop (not `diffeqsolve`). `model_extras` is baked into the score model evaluation. The `sampler_fn` accepts but ignores the override.

### SM SDE/PF path (the only fully dynamic one)

```
build_sampler_fn(model_extras={"cond": ..., "obs_ids": ...})
  └─> solver.get_sampler(model_extras=extras)          # passed at creation
        └─> sample(key, x_init, model_extras=extras)   # ✅ runtime param
              └─> sm_reverse_sde_sampler(model_kwargs=model_extras) # ✅ threaded
                    └─> diffeqsolve(args=model_kwargs)   # ✅ dynamic
                          └─> reverse_drift(t, y, args)  # ✅ reads from args
                                └─> score_model(**args)  # ✅ fully dynamic
```

**Status**: Works correctly. The extras flow through `diffeqsolve(args=...)` and are unpacked in the drift function via `**args`. No closure capture of the actual values.

## What Needs To Be Done

### 1. FM SDE: Remove `model_extras` from `get_sampler()` signature

Currently `model_extras` is passed to `get_sampler()` as a default value but doesn't need to be. The sampler should take `model_extras` as a **required** runtime argument with no default.

**Files**: `sde_solver_fm.py`, `flow_matching.py`

### 2. FM ODE: Use `diffeqsolve(args=model_extras)` instead of `get_vector_field(**model_extras)`

The ODE solver should follow the same pattern: `get_vector_field()` with no kwargs, `diffeqsolve(args=model_extras)` to pass at runtime.

**Files**: `ode_solver.py`, `flow_matching.py`

### 3. EDM: Thread `model_extras` through discrete sampling loop

EDM uses a custom discrete loop, not `diffeqsolve`. Need to understand the EDM solver's sampling implementation and thread `model_extras` through the score model evaluations at runtime.

**Files**: `edm_solver.py`, `diffusion_edm.py`

### 4. Pipeline: `build_sampler_fn()` should not receive `model_extras`

Ideally, the pipeline builds the sampler infrastructure once (solver, step size, method), and `model_extras` is only provided at call time. The pipeline's `get_sampler()` should build the sampler and return a function that takes `model_extras` as its only varying input.

**Files**: `conditional_pipeline.py`, `joint_pipeline.py`, and all `build_sampler_fn()` methods

## Design Note: Static vs Dynamic Kwargs

`get_sampler()` **may** accept static `**kwargs` that get baked into the sampler closure (e.g. solver hyperparameters, configuration flags). This is fine and expected for genuinely static configuration. However, **no current pipeline passes any static model kwargs** — all current `model_extras` (`cond`, `obs_ids`, `cond_ids`, `edge_mask`, etc.) are condition-dependent and must be dynamic. The refactoring should remove `model_extras` from `get_sampler()` entirely, while keeping `**kwargs` available for future truly-static use cases.

## Key Architectural Insight

`ModelWrapper.get_vector_field(**kwargs)` already supports both channels:

```python
def vf(t, x, args):
    args = args if args is not None else {}
    return self(t, x, **args, **kwargs)  # runtime args + static kwargs
```

If called as `get_vector_field()` (no kwargs) and `diffeqsolve(args=model_extras)`, everything flows through `args` at runtime. The `**kwargs` channel remains for truly static parameters.

## Test Coverage Added

- 4 `sample_batched` tests with B=3 in `test_fm_solver.py` (conditional + joint × 2 SDE solvers)
- `sample_batched` in shared `model_test_helpers.py` (covers Flux1, Flux1Joint, Simformer)
- Mock `MockConditionalModel` now broadcasts `cond → obs.shape[0]` to catch batch mismatches
- Warning in `sample()` when `x_o.shape[0] > 1`
- All x_o batch sizes fixed to 1 in `sample()` tests
