# Deprecation Removal Checklist — Next Major Release

All items below are currently **deprecated stubs** or **dead code** left in place for discoverability and migration messaging. They should be deleted in the next major release.

---

## 1. Pipeline Deprecation Stubs (Phase 3)

9 stub classes that raise `RuntimeError` on instantiation. Remove the stub classes, their base classes, and the `__init__.py` exports.

| Stub Class | File | Replacement |
|---|---|---|
| `ConditionalFlowPipeline` | [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py) | `ConditionalPipeline(method=FlowMatchingMethod())` |
| `ConditionalDiffusionPipeline` | same | `ConditionalPipeline(method=DiffusionEDMMethod())` |
| `ConditionalSMPipeline` | same | `ConditionalPipeline(method=ScoreMatchingMethod())` |
| `JointFlowPipeline` | [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py) | `JointPipeline(method=FlowMatchingMethod())` |
| `JointDiffusionPipeline` | same | `JointPipeline(method=DiffusionEDMMethod())` |
| `JointSMPipeline` | same | `JointPipeline(method=ScoreMatchingMethod())` |
| `UnconditionalFlowPipeline` | [unconditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/unconditional_pipeline.py) | `UnconditionalPipeline(method=FlowMatchingMethod())` |
| `UnconditionalDiffusionPipeline` | same | `UnconditionalPipeline(method=DiffusionEDMMethod())` |
| `UnconditionalSMPipeline` | same | `UnconditionalPipeline(method=ScoreMatchingMethod())` |

**Also remove:**
- `_DeprecatedConditionalPipeline` base class in `conditional_pipeline.py`
- `_DeprecatedJointPipeline` base class in `joint_pipeline.py`
- `_DeprecatedUnconditionalPipeline` base class in `unconditional_pipeline.py`

---

## 2. Loss Deprecation Stubs (Phase 4)

9 stub classes in `models/losses/` that raise `RuntimeError` on instantiation.

| Stub Class | File | Replacement |
|---|---|---|
| `ConditionalCFMLoss` | [conditional.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/conditional.py) | `FlowMatchingMethod().build_loss(path)` |
| `ConditionalEDMLoss` | same | `DiffusionEDMMethod().build_loss(path)` |
| `ConditionalSMLoss` | same | `ScoreMatchingMethod().build_loss(path)` |
| `JointCFMLoss` | [joint.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/joint.py) | `FlowMatchingMethod().build_loss(path)` |
| `JointEDMLoss` | same | `DiffusionEDMMethod().build_loss(path)` |
| `JointSMLoss` | same | `ScoreMatchingMethod().build_loss(path)` |
| `UnconditionalCFMLoss` | [unconditional.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/unconditional.py) | `FlowMatchingMethod().build_loss(path)` |
| `UnconditionalEDMLoss` | same | `DiffusionEDMMethod().build_loss(path)` |
| `UnconditionalSMLoss` | same | `ScoreMatchingMethod().build_loss(path)` |

**Also remove:**
- `_DeprecatedLoss` base class in `conditional.py`
- Entire files `conditional.py`, `joint.py`, `unconditional.py` (will be empty after stub removal)
- All 9 loss exports from [models/losses/\_\_init\_\_.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/losses/__init__.py)
- All 9 loss re-exports from [models/\_\_init\_\_.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/__init__.py)

---

## 3. `__init__.py` Export Cleanup

### [recipes/\_\_init\_\_.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/__init__.py)

Remove these imports and `__all__` entries (lines 9-11, 30-38):

```python
# Remove these imports:
from .joint_pipeline import JointDiffusionPipeline, JointFlowPipeline, JointSMPipeline
from .conditional_pipeline import ConditionalFlowPipeline, ConditionalDiffusionPipeline, ConditionalSMPipeline
from .unconditional_pipeline import UnconditionalFlowPipeline, UnconditionalDiffusionPipeline, UnconditionalSMPipeline

# Remove from __all__:
"JointDiffusionPipeline", "JointFlowPipeline", "JointSMPipeline",
"ConditionalFlowPipeline", "ConditionalDiffusionPipeline", "ConditionalSMPipeline",
"UnconditionalFlowPipeline", "UnconditionalDiffusionPipeline", "UnconditionalSMPipeline",
```

### [models/\_\_init\_\_.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/__init__.py)

Remove all 9 loss class imports and `__all__` entries.

---

## 4. Stale Imports in Pipeline Files

These pipeline files still import modules that were only needed by the **old** pipeline implementations. The unified pipelines use `method.build_path()` / `method.build_loss()` and don't directly reference paths, schedulers, or solvers.

| File | Stale Imports |
|---|---|
| [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py) | `AffineProbPath`, `CondOTScheduler`, `ODESolver`, `BaseFmSDESolver`, `ZeroEndsSolver`, `NonSingularSolver`, `EDMPath`, `EDMScheduler`, `VEEdmScheduler`, `VPEdmScheduler`, `EDMSolver`, `SMPath`, `VPSmScheduler`, `VESmScheduler`, `SMSolver`, `SMPFSolver`, `repeat` (einops), `model` (flux1), `optax`, `reduce_on_plateau`, `tqdm`, `partial`, `ocp`, `Union`, `Tuple`, `dist`, `os`, `yaml` |
| [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py) | Same path/scheduler/solver imports, plus `repeat`, `optax`, `reduce_on_plateau`, `tqdm`, `partial`, `ocp`, `os`, `yaml`, `ModelEMA` |
| [unconditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/unconditional_pipeline.py) | Same path/scheduler/solver imports, plus `repeat`, `dist` |

> [!NOTE]
> Verify each import is truly unused before removing — some may be used by the unified pipeline class in the same file. The safest approach is to remove imports, run `pytest`, and see if anything breaks.

---

## 5. `old/` Directory

Files moved here during refactoring. Delete the entire directory once confident:

| File | Origin |
|---|---|
| `old/continuous_loss.py` | `flow_matching/loss/continuous_loss.py` — `ContinuousFMLoss` base class (Phase 4) |
| `old/test_conditional_pipeline.py` | Old pipeline tests (Phase 3) |
| `old/test_joint_pipeline.py` | Old pipeline tests (Phase 3) |
| `old/test_unconditional_pipeline.py` | Old pipeline tests (Phase 3) |
| `old/test_pipelines_models.py` | Old pipeline-model integration tests (Phase 3) |

---

## 6. Tests to Remove

Tests that verify stubs raise `RuntimeError` — no longer needed once stubs are deleted:

| Test File | Tests |
|---|---|
| `tests/models/losses/test_conditional_loss.py` | 3 RuntimeError tests → delete file |
| `tests/models/losses/test_joint_loss.py` | 3 RuntimeError tests → delete file |
| `tests/models/losses/test_unconditional_loss.py` | 3 RuntimeError tests → delete file |
| `tests/flow_matching/loss/test_continuous_loss.py` | 2 ImportError tests → delete file |
| `tests/recipes/test_deprecated_pipelines.py` | RuntimeError tests for pipeline stubs → delete file (if exists) |

> [!TIP]
> After removing all stubs, tests, and exports, the `models/losses/` directory may become empty. Consider removing it entirely or repurposing it.
