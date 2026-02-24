# Phase 4: Simplify Losses

## Goal

Remove 9 mode-specific loss wrappers from `models/losses/`. Core losses are already handled by the `GenerativeMethod` strategies via `method.build_loss(path)`.

---

## Current state

After Phase 2B, each strategy returns a first-class loss object:

| Strategy | `build_loss(path)` returns | Location |
|---|---|---|
| `FlowMatchingMethod` | `FMLoss(path)` | `flow_matching/loss/fm_loss.py` |
| `DiffusionEDMMethod` | `EDMLoss(path)` | `diffusion/loss/edm_loss.py` |
| `ScoreMatchingMethod` | `SMLoss(path)` | `diffusion/loss/sm_loss.py` |

The old loss classes in `models/losses/` are **only used by the old pipeline classes** (which remain until Phase 3). Therefore Phase 4 can only happen **after Phase 3** migrates all pipelines to use the new unified classes.

> [!WARNING]
> This phase depends on Phase 3 completing first, or can be combined with Phase 3.

---

## Files to modify

### [models/losses/conditional.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/conditional.py)

Replace 3 classes with deprecation stubs:

| Class | Replacement |
|---|---|
| `ConditionalCFMLoss` | `FlowMatchingMethod().build_loss(path)` |
| `ConditionalEDMLoss` | `DiffusionEDMMethod().build_loss(path)` |
| `ConditionalSMLoss` | `ScoreMatchingMethod().build_loss(path)` |

### [models/losses/joint.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/joint.py)

Replace 3 classes:

| Class | Replacement |
|---|---|
| `JointCFMLoss` | `FlowMatchingMethod().build_loss(path)` |
| `JointEDMLoss` | `DiffusionEDMMethod().build_loss(path)` |
| `JointSMLoss` | `ScoreMatchingMethod().build_loss(path)` |

### [models/losses/unconditional.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/unconditional.py)

Replace 3 classes:

| Class | Replacement |
|---|---|
| `UnconditionalCFMLoss` | `FlowMatchingMethod().build_loss(path)` |
| `UnconditionalEDMLoss` | `DiffusionEDMMethod().build_loss(path)` |
| `UnconditionalSMLoss` | `ScoreMatchingMethod().build_loss(path)` |

---

## Stub pattern

```python
class ConditionalEDMLoss:
    """Deprecated. Use ``DiffusionEDMMethod().build_loss(path)`` or
    ``ConditionalPipeline(method=DiffusionEDMMethod(), ...)`` instead."""

    def __init__(self, *args, **kwargs):
        raise DeprecationError(
            "ConditionalEDMLoss has been removed. "
            "Use DiffusionEDMMethod().build_loss(path) or "
            "ConditionalPipeline(method=DiffusionEDMMethod(), ...) instead."
        )
```

---

## Tests

Update `tests/models/losses/` to test that old names raise `DeprecationError` on instantiation.

## Extra: Remove `ContinuousFMLoss`

> [!IMPORTANT]
> After all old loss subclasses (`ConditionalCFMLoss`, `JointCFMLoss`, `UnconditionalCFMLoss`) are
> removed (this phase), `ContinuousFMLoss` in `flow_matching/loss/continuous_loss.py` becomes unused.
> Delete `continuous_loss.py`, remove its export from `flow_matching/loss/__init__.py`, and update
> `tests/flow_matching/loss/test_continuous_loss.py` accordingly.

## Verification

```bash
python -m pytest tests/models/losses/ tests/recipes/ -x --tb=short
```
