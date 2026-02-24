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
Phase 1 (DONE) → Phase 2 → Phase 4 → Phase 3
                             ↑
                    (must migrate Flux1/Simformer
                     before deprecating old classes)
```

### Phase 1 (DONE): Core module

Created `GenerativeMethod` ABC and 3 strategies in `src/gensbi/core/`. Each strategy encapsulates:
- `build_path(config)` — creates the probability path
- `build_loss(path)` — creates the training loss
- `prepare_batch(key, x_1, path)` — samples noise/time, returns uniform `(x_0, x_1, t_or_sigma)`
- `build_sampler_fn(model, path, extras, **kwargs)` — builds inference sampler
- `sample_init(key, shape, path)` — samples initial noise

### Phase 2: Create unified pipelines

Add `ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline` **alongside** old classes. Each is model-agnostic — users supply any conforming model + a `method` argument. See `phase2_handout.md`.

### Phase 4: Migrate model-specific pipelines, then deprecate

Migrate `Flux1FlowPipeline` etc. from inheriting `ConditionalFlowPipeline` to inheriting `ConditionalPipeline(method=FlowMatchingMethod())`. Once done, old generic classes become deprecation stubs. See `phase4_handout.md`.

### Phase 3: Stub out loss wrappers

Now that nothing uses the old loss classes, replace them with deprecation stubs. See `phase3_handout.md`.

---

## Key Design Principles

1. **Model-agnostic pipelines.** The user provides their own model. The pipeline doesn't care which model it is, as long as it conforms to the wrapper interface (`ConditionalWrapper`, `JointWrapper`, `UnconditionalWrapper`). This is the core value proposition — it was never about the 3 built-in models, but about enabling any model.

2. **Strategy pattern, not inheritance.** The method (flow/diffusion/SM) is a composed object, not a subclass axis. One pipeline class works for all methods.

3. **Old classes stay until consumers migrate.** Model-specific pipelines (Flux1, Simformer) inherit from the old generic classes. We can't deprecate the old classes until Phase 4 migrates them to the new base.

4. **Uniform loss interface.** All `build_loss(path)` return objects with identical signature: `(model, batch, condition_mask=None, model_extras=None) → scalar`. This required adding `_FMLoss` wrapper since `ContinuousFMLoss` uses `**kwargs`.

5. **Uniform batch format.** All `prepare_batch` return `(x_0, x_1, t_or_sigma)`. All `path.sample` take `(x_0, x_1, t_or_sigma)`. No key-passing to loss functions.

---

## Failure Modes and Lessons Learned

### 1. Inheritance chains block deprecation

**What happened:** Phase 2 initially tried to replace old classes with deprecation stubs. But `Flux1FlowPipeline(ConditionalFlowPipeline)` inherits from the old class — making it a stub breaks all model-specific pipelines.

**Lesson:** Always check the full inheritance tree before removing a class. Use `grep -r "class.*MyClass" src/` to find all subclasses.

**Fixed approach:** Append new classes alongside old ones. Deprecate only after Phase 4 migrates consumers.

### 2. Loss interface mismatch

**What happened:** `ContinuousFMLoss.__call__(vf, batch, **kwargs)` has a fundamentally different signature from `_EDMLoss.__call__(model, batch, condition_mask, model_extras)`. The unified pipeline needs to call all losses the same way.

**Lesson:** When creating strategy patterns, ensure all strategy implementations expose the same interface. Wrapper adapters are cheap.

**Fixed approach:** `_FMLoss` adapter wraps `ContinuousFMLoss` with the uniform interface.

### 3. Mode-specific differences are subtle

The three pipeline modes (conditional/joint/unconditional) differ in non-obvious ways:

| Aspect | Conditional | Joint | Unconditional |
|---|---|---|---|
| `batch` from dataset | `(obs, cond)` tuple | `x_1` (concatenated) | `obs` (not a tuple) |
| IDs | `obs_ids`, `cond_ids` | `node_ids`, `obs_ids`, `cond_ids` | `obs_ids` only |
| `condition_mask` in loss | not used | sampled per batch | `zeros` (always) |
| `model_extras` in loss | `{cond, obs_ids, cond_ids}` | `{node_ids}` | `{obs_ids}` |
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

## Current Codebase State (as of Phase 1C completion)

All Phase 1/1B/1C changes are **committed**. The codebase is clean. Key files created:

- `src/gensbi/core/__init__.py`
- `src/gensbi/core/generative_method.py` — `GenerativeMethod` ABC
- `src/gensbi/core/flow_matching.py` — `FlowMatchingMethod`
- `src/gensbi/core/diffusion_edm.py` — `DiffusionEDMMethod`
- `src/gensbi/core/score_matching.py` — `ScoreMatchingMethod`
- `tests/core/test_generative_method.py`

Key modifications:
- `src/gensbi/diffusion/path/edm_path.py` — `EDMPath.sample(x_0, x_1, sigma)` takes noise directly
- `src/gensbi/diffusion/path/sm_path.py` — `SMPath.sample(x_0, x_1, t)` takes noise directly

---

## Test Verification Command

```bash
# Always use this pattern for tests:
eval "$(conda shell.bash hook)" && conda deactivate && conda deactivate && conda activate gensbi
pytest tests/ -x --tb=short -q
```
