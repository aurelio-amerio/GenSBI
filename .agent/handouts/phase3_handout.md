# Phase 3: Simplify Model-Specific Pipelines & Deprecate Old Classes

## Goal

1. Migrate model-specific pipelines to inherit from the new unified classes
2. Replace old generic pipeline classes with deprecation stubs
3. Deduplicate `parse_training_config` into shared utility

---

## Current inheritance

```
Flux1FlowPipeline          → ConditionalFlowPipeline
Flux1DiffusionPipeline     → ConditionalDiffusionPipeline
Flux1SMPipeline            → ConditionalSMPipeline

Flux1JointFlowPipeline     → JointFlowPipeline
Flux1JointDiffusionPipeline → JointDiffusionPipeline
Flux1JointSMPipeline       → JointSMPipeline

SimformerFlowPipeline      → JointFlowPipeline
SimformerSMPipeline        → JointSMPipeline
SimformerDiffusionPipeline → JointDiffusionPipeline
```

## Target inheritance

All model-specific pipelines become thin subclasses that:
1. Create the appropriate model
2. Choose the right `GenerativeMethod`
3. Call `super().__init__(..., method=Method(), ...)`

```
Flux1FlowPipeline          → ConditionalPipeline(method=FlowMatchingMethod())
Flux1DiffusionPipeline     → ConditionalPipeline(method=DiffusionEDMMethod(sde=...))
Flux1SMPipeline            → ConditionalPipeline(method=ScoreMatchingMethod(sde_type=...))

Flux1JointFlowPipeline     → JointPipeline(method=FlowMatchingMethod())
Flux1JointDiffusionPipeline → JointPipeline(method=DiffusionEDMMethod(sde=...))
Flux1JointSMPipeline       → JointPipeline(method=ScoreMatchingMethod(sde_type=...))

SimformerFlowPipeline      → JointPipeline(method=FlowMatchingMethod())
SimformerSMPipeline        → JointPipeline(method=ScoreMatchingMethod(sde_type=...))
SimformerDiffusionPipeline → JointPipeline(method=DiffusionEDMMethod(sde=...))
```

---

## File changes

### [flux1.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/flux1.py)

Collapse 3 classes into thin subclasses of `ConditionalPipeline`. Each class:
- Creates default `Flux1Params` if not provided
- Creates a `Flux1` model
- Picks the right `GenerativeMethod`
- Calls `super().__init__(..., method=..., id_embedding_strategy=params.id_embedding_strategy, ...)`
- Provides `init_pipeline_from_config` classmethod
- Provides `get_default_params` classmethod

```python
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod
from .conditional_pipeline import ConditionalPipeline

class Flux1FlowPipeline(ConditionalPipeline):
    def __init__(self, train_dataset, val_dataset, dim_obs, dim_cond,
                 ch_obs=1, ch_cond=1, params=None, training_config=None):
        if params is None:
            params = Flux1Params(...)
        model = Flux1(params)
        super().__init__(
            model=model, ...,
            method=FlowMatchingMethod(),
            id_embedding_strategy=(params.id_embedding_kind, params.id_embedding_kind),
        )

class Flux1DiffusionPipeline(ConditionalPipeline):
    def __init__(self, ..., sde="EDM", ...):
        ...
        super().__init__(..., method=DiffusionEDMMethod(sde=sde), ...)

class Flux1SMPipeline(ConditionalPipeline):
    def __init__(self, ..., sde_type="VP", ...):
        ...
        super().__init__(..., method=ScoreMatchingMethod(sde_type=sde_type), ...)
```

### [flux1joint.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/flux1joint.py)

Same pattern → subclasses of `JointPipeline`. 3 classes.

### [simformer.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/simformer.py)

Same pattern → subclasses of `JointPipeline`. 3 classes.

> [!IMPORTANT]
> Simformer has `edge_mask` in `__init__` and passes it as a `model_extra` in `get_sampler`/`get_loss_fn`. This may require either:
> - Overriding `get_sampler` to add `edge_mask` to `model_extras`
> - Storing `edge_mask` and having the unified `JointPipeline` support extra model kwargs

---

## Deprecation stubs

After migration, replace old generic pipeline classes with stubs:

### [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py)

Replace `ConditionalFlowPipeline`, `ConditionalDiffusionPipeline`, `ConditionalSMPipeline`:

```python
class ConditionalFlowPipeline:
    def __init__(self, *a, **kw):
        raise DeprecationError(
            "Use ConditionalPipeline(method=FlowMatchingMethod(), ...) instead."
        )
```

### [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py)

Replace `JointFlowPipeline`, `JointDiffusionPipeline`, `JointSMPipeline`.

### [unconditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/unconditional_pipeline.py)

Replace `UnconditionalFlowPipeline`, `UnconditionalDiffusionPipeline`, `UnconditionalSMPipeline`.

> [!NOTE]
> Once old generic pipeline classes are deprecated, Phase 4 can proceed to remove
> the old mode-specific loss classes in `models/losses/` and `ContinuousFMLoss`
> from `flow_matching/loss/continuous_loss.py`.

---

## Deduplication: `parse_training_config`

Each model-specific file currently has a `parse_training_config` function with near-identical logic. Extract into [recipes/utils.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/utils.py).

---

## Tests

- All existing pipeline tests must pass (public API unchanged)
- Model-specific pipeline tests must pass (same behavior, different inheritance)
- Old generic class names raise `DeprecationError`

## Verification

```bash
python -m pytest tests/recipes/ tests/core/ -x --tb=short
```
