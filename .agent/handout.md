# EDM Ablation Sampler Fix — Verification Handout

## Problem

When using `pipeline_edm.sample(..., solver_scheduler=ve_scheduler)`, the model trained with EDM preconditioning produced completely wrong results. The root cause: [edm_ablation_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_samplers.py#160-337) called `sde.denoise()` on the **sampling scheduler** (VP/VE), which applied that scheduler's preconditioning ([c_skip](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#675-678), [c_in](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#193-209), [c_out](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#679-682), [c_noise](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#612-615)) instead of the **training scheduler's** (EDM). Per the EDM paper, preconditioning must always match training — only the sampling dynamics (timesteps, [sigma](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#74-90), `s`, scaling) should change.

## What We Implemented

Decoupled preconditioning from sampling dynamics in [edm_ablation_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_samplers.py#160-337), following the reference EDM [ablation_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/wip-score/edm-main%202/generate.py#66-177) from [generate.py](file:///lhome/ific/a/aamerio/data/github/GenSBI/wip-score/edm-main%202/generate.py):

- **[edm_samplers.py](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_samplers.py)**: Added `denoise_scheduler` parameter (handles [denoise()](file:///lhome/ific/a/aamerio/data/github/GenSBI/tests/diffusion/solver/test_edm_samplers.py#24-27) = preconditioning). Renamed [sde](file:///lhome/ific/a/aamerio/data/github/GenSBI/tests/diffusion/solver/test_edm_samplers.py#38-41) → `sampling_scheduler` in signature (aliased to [sde](file:///lhome/ific/a/aamerio/data/github/GenSBI/tests/diffusion/solver/test_edm_samplers.py#38-41) locally for readability). Only `denoise_scheduler.denoise()` is called for model evaluation; all other scheduler methods ([sigma](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#74-90), `s`, [timesteps](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/path/scheduler/edm.py#55-73), etc.) use the sampling scheduler.
- **[edm_solver.py](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_solver.py)**: Added `from functools import partial`. In [get_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/recipes/conditional_pipeline.py#235-288), when ablation sampler is selected, binds `denoise_scheduler=self.path.scheduler` (the training scheduler) via `partial`.
- **[test_edm_samplers.py](file:///lhome/ific/a/aamerio/data/github/GenSBI/tests/diffusion/solver/test_edm_samplers.py)**: New tests for [edm_ablation_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_samplers.py#160-337) with VP/VE sampling + EDM denoising, determinism, stochasticity, and Euler method.

## Tests to Run

All commands assume:
```bash
mamba deactivate && mamba deactivate && mamba activate gensbi
```

### 1. Unit tests — sampler functions

```bash
pytest tests/diffusion/solver/test_edm_samplers.py -x --tb=short -v
```

Validates: [edm_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/wip-score/edm-main%202/generate.py#25-61) (existing) + [edm_ablation_sampler](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/diffusion/solver/edm_samplers.py#160-337) (new tests).

### 2. Pipeline tests — EDM solver across all pipeline types

```bash
pytest tests/recipes/test_solver_edm_pipelines.py -x --tb=short -v
```

Validates: sampling with EDM/VP/VE schedulers through [ConditionalPipeline](file:///lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/recipes/conditional_pipeline.py#102-322), `UnconditionalPipeline`, `JointPipeline`, including cross-scheduler swaps.

### 3. Full test suite — regression check

```bash
pytest tests/ -x --tb=short
```
