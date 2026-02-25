# GenSBI Pipeline Consolidation — Global Overview

## What We Want to Achieve

The GenSBI codebase has a **combinatorial explosion** of pipeline and loss classes: 3 inference modes × 3 generative methods = 9 generic pipelines + 9 loss wrappers + 9 model-specific pipelines = **27 classes** with massive code duplication.

**Target:** Collapse these into **3 mode pipelines** (`ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline`) + **3 strategy objects** (`FlowMatchingMethod`, `DiffusionEDMMethod`, `ScoreMatchingMethod`), composed via the strategy pattern. Model-specific pipelines become thin subclasses.

| Metric | Before | After |
|---|---|---|
| Generic pipeline classes | 9 | 3 |
| Loss wrapper classes | 9 | 0 (strategies own losses) |
| Lines of pipeline code | ~5994 | ~2431 (-59%) |

---

## How to Do It

### Phase dependency graph

```
Phase 1 (DONE) → Phase 2 (DONE) → Phase 2B (DONE) → Phase 3 (DONE) → Phase 4
```

### Phase 1 (DONE): Core module

Created `GenerativeMethod` ABC and 3 strategies in `src/gensbi/core/`. Each strategy encapsulates:
- `build_path(config)` — creates the probability path
- `build_loss(path)` — creates the training loss
- `prepare_batch(key, x_1, path)` — samples noise/time, returns uniform `(x_0, x_1, t_or_sigma)`
- `build_sampler_fn(model, path, extras, **kwargs)` — builds inference sampler
- `sample_init(key, shape, path)` — samples initial noise

### Phase 2 (DONE): Create unified pipelines

Added `ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline` **alongside** old classes. Each is model-agnostic — users supply any conforming model + a `method` argument. 30 parameterized tests added.

### Phase 2B (DONE): Promote loss classes

Moved private loss wrappers to first-class citizens in canonical locations:

| Class | Location |
|---|---|
| `FMLoss` | `flow_matching/loss/fm_loss.py` |
| `EDMLoss` | `diffusion/loss/edm_loss.py` |
| `SMLoss` | `diffusion/loss/sm_loss.py` |

### Phase 3 (DONE): Migrate model-specific pipelines, then deprecate

Migrated `Flux1FlowPipeline` etc. from inheriting `ConditionalFlowPipeline` to inheriting `ConditionalPipeline(method=FlowMatchingMethod())`. Old generic classes are now deprecation stubs that raise `RuntimeError`. Fixed `node_ids` bug in `UnconditionalPipeline.get_loss_fn()`. Tests reorganized: pipeline tests use mock models, model integration tests split into 3 files. See `phase3_handout.md`.

### Phase 4: Stub out loss wrappers

Now that nothing uses the old loss classes, replace them with deprecation stubs. See `phase4_handout.md`.

---

## Key Design Principles

1. **Model-agnostic pipelines.** The user provides their own model. The pipeline doesn't care which model it is, as long as it conforms to the wrapper interface (`ConditionalWrapper`, `JointWrapper`, `UnconditionalWrapper`). This is the core value proposition — it was never about the 3 built-in models, but about enabling any model.

2. **Strategy pattern, not inheritance.** The method (flow/diffusion/SM) is a composed object, not a subclass axis. One pipeline class works for all methods.

3. **Old classes stay until consumers migrate.** Model-specific pipelines (Flux1, Simformer) inherit from the old generic classes. We can't deprecate the old classes until Phase 4 migrates them to the new base.

4. **Uniform loss interface.** All `build_loss(path)` return objects with identical signature: `(model, batch, condition_mask=None, model_extras=None) → scalar`. `FMLoss`, `EDMLoss`, `SMLoss` all live in their canonical `loss/` subpackages.

5. **Uniform batch format.** All `prepare_batch` return `(x_0, x_1, t_or_sigma)`. All `path.sample` take `(x_0, x_1, t_or_sigma)`. No key-passing to loss functions.

---

## Failure Modes and Lessons Learned

### 1. Inheritance chains block deprecation

**What happened:** Phase 2 initially tried to replace old classes with deprecation stubs. But `Flux1FlowPipeline(ConditionalFlowPipeline)` inherits from the old class — making it a stub breaks all model-specific pipelines.

**Lesson:** Always check the full inheritance tree before removing a class. Use `grep -r "class.*MyClass" src/` to find all subclasses.

**Fixed approach:** Append new classes alongside old ones. Deprecate only after Phase 4 migrates consumers.

### 2. Loss interface mismatch

**What happened:** `ContinuousFMLoss.__call__(vf, batch, **kwargs)` has a fundamentally different signature from `EDMLoss.__call__(model, batch, condition_mask, model_extras)`. The unified pipeline needs to call all losses the same way.

**Lesson:** When creating strategy patterns, ensure all strategy implementations expose the same interface. Wrapper adapters are cheap.

**Fixed approach:** `FMLoss` (in `flow_matching/loss/fm_loss.py`) implements the loss directly rather than wrapping `ContinuousFMLoss`.

### 2b. Model calling convention: named args matter

**What happened:** `ContinuousFMLoss` calls `vf(x_t, t, **kwargs)` — positional args with `x_t` first. But model wrappers (`ConditionalWrapper`, etc.) expect `(t, obs, ...)` — `t` first. This caused a shape mismatch `(1, 32, 1)` vs `(32, 2, 2)` that was hard to debug.

**Lesson:** When losses call models, always use **named arguments** `model(obs=x_t, t=t, **model_extras)`. This matches the convention used by the EDM/SM scheduler loss functions and avoids positional arg order bugs.

### 2c. `condition_mask` serves double duty in joint models

**What happened:** Joint models need `condition_mask` both for (1) masking `x_t` (setting conditioned variables to clean data) and (2) as a model input. The `_FMLoss` initially only did the x_t masking but didn't pass the mask to the model, causing `missing argument` errors.

**Lesson:** `condition_mask` must be included in `model_extras` so it gets passed through to the model. The EDM/SM scheduler loss functions pass `**model_extras` to the model, so if `condition_mask` isn't in there, the model never sees it.

### 3. Mode-specific differences are subtle

The three pipeline modes (conditional/joint/unconditional) differ in non-obvious ways:

| Aspect | Conditional | Joint | Unconditional |
|---|---|---|---|
| `batch` from dataset | `(obs, cond)` tuple | `x_1` (concatenated) | `obs` (not a tuple) |
| IDs | `obs_ids`, `cond_ids` | `node_ids`, `obs_ids`, `cond_ids` | `obs_ids` only |
| `condition_mask` in loss | not used | sampled per batch | `zeros` (always) |
| `model_extras` in loss | `{cond, obs_ids, cond_ids}` | `{node_ids}` | `{node_ids}` |
| `model_extras` in sampler | `{cond, obs_ids, cond_ids}` | `{cond, obs_ids, cond_ids}` | `{obs_ids}` |
| `x_init` shape in sampler | `(n, dim_obs, ch)` | `(n, dim_obs, ch)` | `(n, dim_obs, ch)` |
| `get_sampler` takes `x_o`? | yes | yes | no |
| Wrapper | `ConditionalWrapper` | `JointWrapper` | `UnconditionalWrapper` |

**Lesson:** Study every method of every variant before writing the unified class. The `get_loss_fn` differences are where most bugs will hide.

### 4. Environment activation pitfalls

`mamba run -n gensbi` and `conda run -n gensbi` can silently use the wrong environment. The reliable pattern is:

```bash
eval "$(conda shell.bash hook)" && conda deactivate && conda deactivate && conda activate gensbi
```

Then run `pytest` directly. This applies to **all** test verification.

### 5. Partial implementations create confusion

**What happened:** Starting Phase 2 implementation before the plan was approved led to partial changes that had to be reverted.

**Lesson:** Get plan approval first. Write detailed handouts. Work one phase at a time with clean start/end states. Commit after each phase.

---

## Current Codebase State (as of Phase 3 completion)

All Phase 1/1B/1C/2/2B/3 changes are applied. Key files:

**Core strategies (`src/gensbi/core/`):**
- `generative_method.py` — `GenerativeMethod` ABC
- `flow_matching.py` — `FlowMatchingMethod` (imports `FMLoss`)
- `diffusion_edm.py` — `DiffusionEDMMethod` (imports `EDMLoss`)
- `score_matching.py` — `ScoreMatchingMethod` (imports `SMLoss`)

**Canonical loss classes:**
- `src/gensbi/flow_matching/loss/fm_loss.py` — `FMLoss`
- `src/gensbi/diffusion/loss/edm_loss.py` — `EDMLoss`
- `src/gensbi/diffusion/loss/sm_loss.py` — `SMLoss`

**Unified pipelines (old generic classes are deprecation stubs):**
- `src/gensbi/recipes/conditional_pipeline.py` — `ConditionalPipeline` + 3 stubs
- `src/gensbi/recipes/joint_pipeline.py` — `JointPipeline` + 3 stubs
- `src/gensbi/recipes/unconditional_pipeline.py` — `UnconditionalPipeline` + 3 stubs

**Model-specific pipelines (migrated to unified base):**
- `src/gensbi/recipes/flux1.py` — `Flux1FlowPipeline(ConditionalPipeline)`
- `src/gensbi/recipes/flux1joint.py` — `Flux1JointFlowPipeline(JointPipeline)`
- `src/gensbi/recipes/simformer.py` — `SimformerFlowPipeline(JointPipeline)`

**Tests:** 481 passing.

---

## Test Verification Command

```bash
# Always use this pattern for tests:
mamba deactivate && mamba deactivate && mamba activate gensbi
pytest tests/ -x --tb=short -q
```

See `.agent/handouts/best_practices.md` for more details.
