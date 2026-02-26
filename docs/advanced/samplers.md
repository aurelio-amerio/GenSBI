# Samplers and Solvers

GenSBI supports **Flow Matching** and **Diffusion** as generative methods. Diffusion comes in two flavors: **EDM** (modern, learns a denoiser) and **Score Matching** (classical, learns the score function). Each has its own set of solvers. This page explains how to choose between them and how to override the default solver when sampling.

## Generative Methods

A `GenerativeMethod` encapsulates the mathematical framework (path, loss, solver) used by a pipeline. You pass it when creating a unified pipeline:

```python
from gensbi.recipes import ConditionalPipeline  # or JointPipeline, UnconditionalPipeline
from gensbi.core import FlowMatchingMethod

pipeline = ConditionalPipeline(
    model, train_ds, val_ds,
    dim_obs=3, dim_cond=3,
    method=FlowMatchingMethod(),
)
```

### Available Methods

| Method | Import | Description |
|--------|--------|-------------|
| `FlowMatchingMethod()` | `from gensbi.core import FlowMatchingMethod` | **Recommended.** Learns a velocity field; integrates forward from noise ($t{=}0$) to data ($t{=}1$). |
| `DiffusionEDMMethod()` | `from gensbi.core import DiffusionEDMMethod` | EDM diffusion (Karras et al., 2022). Learns a **denoiser** $D_\theta(x;\sigma)$ in $\sigma$-space. Recommended diffusion variant. |
| `DiffusionEDMMethod(sde="VP")` | | EDM with Variance Preserving scheduler. |
| `DiffusionEDMMethod(sde="VE")` | | EDM with Variance Exploding scheduler. |
| `ScoreMatchingMethod()` | `from gensbi.core import ScoreMatchingMethod` | Classical diffusion (Song et al., 2021). Learns the **score function** $\nabla \log p_t(x)$ with VP SDE (default). |
| `ScoreMatchingMethod(sde_type="VE")` | | Score matching with VE SDE. |

---

## Solvers per Method

Each method has a default solver and one or more alternatives. You can override the solver at sample time without retraining.

### Flow Matching Solvers

```{admonition} Import
:class: tip
`from gensbi.flow_matching.solver import ODESolver, ZeroEndsSolver, NonSingularSolver`
```

| Solver | Type | Default? | Description |
|--------|------|----------|-------------|
| `ODESolver` | Deterministic ODE | ✅ Yes | Standard ODE integration (Euler, Dopri5). Fast and robust. |
| `ZeroEndsSolver` | Stochastic SDE | No | SDE from arXiv:2410.02217 (Tab. 1). Diffusion vanishes at both time endpoints. Can improve sample diversity. |
| `NonSingularSolver` | Stochastic SDE | No | SDE from arXiv:2410.02217 (Tab. 1). Non-singular diffusion coefficient. |

**SDE solver kwargs** (required for `ZeroEndsSolver` and `NonSingularSolver`):

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu0` | `jnp.ndarray` | Prior mean, shape `(dim_obs, ch_obs)` |
| `sigma0` | `jnp.ndarray` | Prior std, shape `(dim_obs, ch_obs)` |
| `alpha` | `float` | Diffusion strength parameter (e.g., `1.0`) |

```{note}
For normalized data (zero mean, unit variance), use `mu0 = jnp.zeros((dim_obs, ch_obs))` and `sigma0 = jnp.ones((dim_obs, ch_obs))`.
```

### EDM Diffusion Solver

EDM and Score Matching are both diffusion-based methods, but use different solvers.

```{admonition} Import
:class: tip
`from gensbi.diffusion.solver import EDMSolver`
```

| Solver | Type | Default? | Description |
|--------|------|----------|-------------|
| `EDMSolver` | Deterministic | ✅ Yes | Stochastic denoising sampler from Karras et al., 2022. |

The EDM solver is always used for `DiffusionEDMMethod`. The scheduler variant (EDM, VP, VE) is set at **pipeline creation time** via `DiffusionEDMMethod(sde=...)`, not at sample time.

### Score Matching Solvers

```{admonition} Import
:class: tip
`from gensbi.diffusion.solver import SMSolver, SMPFSolver`
```

| Solver | Type | Default? | Description |
|--------|------|----------|-------------|
| `SMSolver` | Stochastic (reverse SDE) | ✅ Yes | Standard reverse-SDE sampler from Song et al., 2021. |
| `SMPFSolver` | Deterministic (probability flow ODE) | No | Probability flow ODE. Deterministic samples from the same learned score. |

**No additional kwargs needed** — pass `{}` as the kwargs dict.

---

## How to Override the Solver

Pass a `solver` tuple of `(SolverClass, kwargs_dict)` to `pipeline.sample()`:

```python
# Default (no solver argument needed)
samples = pipeline.sample(key, x_o, nsamples=10_000)

# Override with ZeroEndsSolver (flow matching)
from gensbi.flow_matching.solver import ZeroEndsSolver

solver_kwargs = {
    "mu0": jnp.zeros((dim_obs, 1)),
    "sigma0": jnp.ones((dim_obs, 1)),
    "alpha": 1.0,
}
samples = pipeline.sample(key, x_o, nsamples=10_000, solver=(ZeroEndsSolver, solver_kwargs))

# Override with SMPFSolver (score matching)
from gensbi.diffusion.solver import SMPFSolver

samples = pipeline.sample(key, x_o, nsamples=10_000, solver=(SMPFSolver, {}))
```

```{important}
The solver override only affects sampling — it does **not** require retraining. You can train once and compare results from different solvers.
```

---

## Quick Reference

| Method | Default Solver | Alternative Solvers |
|--------|---------------|---------------------|
| `FlowMatchingMethod` | `ODESolver` (ODE) | `ZeroEndsSolver` (SDE), `NonSingularSolver` (SDE) |
| `DiffusionEDMMethod` | `EDMSolver` | — |
| `ScoreMatchingMethod` | `SMSolver` (reverse SDE) | `SMPFSolver` (probability flow ODE) |

For working examples demonstrating these solvers, see:
- [Flow Matching examples](../examples/flux1_flow_pipeline.py) — includes `ZeroEndsSolver` section
- [Score Matching examples](../examples/flux1_sm_pipeline.py) — includes `SMPFSolver` section
- [Unified pipeline examples](../examples/conditional_pipeline.py) — shows method + solver selection
