# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GenSBI is a JAX / Flax NNX library for simulation-based inference with generative models
(flow matching, EDM diffusion, score matching, discrete normalizing flows). Package source is
`src/gensbi`, Python >= 3.12, built with `uv_build`. Companion paper: arXiv 2605.27499.
Example notebooks live in the separate `GenSBI-examples` repo, not here.

## Commands

Everything runs through `uv` (the `.venv` is managed by it).

```bash
uv sync --group dev            # lint + test + docs + notebooks groups (CPU)
uv sync --group cuda13full     # same plus gensbi[cuda13,examples]
```

Tests (`pyproject.toml` already sets `-n 2`, `JAX_PLATFORMS=cpu`, 8 fake host devices for the
sharding tests, and XLA flags for CPU-consistent numerics; do not override these casually):

```bash
uv run pytest                                              # whole suite
uv run pytest -m "not slow and not experimental and not extraslow"   # CI "fast" tier
uv run pytest -m slow                                      # CI "slow" tier
uv run pytest -m experimental                              # CI "experimental" tier
uv run pytest tests/core/test_ode_solver.py::test_name -n 0           # one test, no xdist
uv run pytest --cov=gensbi --cov-report=term-missing
```

`extraslow` (end-to-end, e.g. `tests/normalizing_flows/test_flow_pipeline_e2e.py`) is never run
in CI. CI also re-runs the fast tier with `jax==0.11.1` pinned: new code must work at the JAX
floor declared in `pyproject.toml`, not only at the latest release.

Lint (CI hard gate is the first line; the second is warnings only, 127-char lines):

```bash
uv run flake8 . --exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
uv run flake8 . --exclude=.venv --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Docs (Sphinx + MyST-NB + sphinx-autoapi over `src/gensbi`; notebooks are **not** executed at build):

```bash
cd docs && uv run make html        # output in docs/_build/html
cd docs && make download           # pulls notebooks from GenSBI-examples into docs/notebooks etc.
cd docs && make preview            # download + html + http.server on :8000
```

Notebooks under `docs/notebooks`, `docs/getting_started/my_first_model.ipynb` and
`docs/advanced/methods_and_samplers.ipynb` are downloaded copies; fix them upstream in
GenSBI-examples, not here.

## Architecture

The package is layered. The math primitives import nothing above them; `core/` is the one
exception to strict layering, since `DiffusionEDMMethod` / `ScoreMatchingMethod` reuse the
path builders in `recipes/utils.py`.

1. **Math primitives.** `flow_matching/` (probability paths + schedulers, ODE solvers, losses),
   `diffusion/` (EDM and score-matching paths, schedulers, SDE solvers), `normalizing_flows/`
   (pure bijections: `Chain`, `Permutation`, `Affine`, `RQSpline`, ...). `solver.py` is the
   abstract `Solver`.
2. **`core/` — `GenerativeMethod` strategy.** `FlowMatchingMethod`, `DiffusionEDMMethod`,
   `ScoreMatchingMethod` each implement `build_path`, `build_loss`, `prepare_batch`
   (always returns `(x_0, x_1, t_or_sigma)`), `build_solver`, `sample_init`. They are
   lazy-loaded from `gensbi.core` to avoid import cycles with the solver subclasses.
3. **`models/`.** NNX architectures: `Flux1` (conditional), `Flux1Joint`, `Simformer` (joint),
   `MAFlow` / `TarFlow` (discrete autoregressive flows), `HealSwinEncoder`. `models/core/` holds
   shared patching, tokenizers and stats. `models/wrappers/` (`ConditionalWrapper`,
   `JointWrapper`, `UnconditionalWrapper`, all subclasses of `utils.model_wrapping.ModelWrapper`)
   adapt a model to the solver interface (`get_vector_field`, `get_divergence`).
4. **`recipes/` — pipelines.** `AbstractPipeline` (`recipes/pipeline.py`) owns the training
   loop, optimizer, EMA (`ModelEMA`), orbax checkpointing, safetensors export/import, sharding
   placement and `sample` / `sample_batched` / `log_prob`. The mode pipelines
   `ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline` take `(model, method)` and
   are model-agnostic. The named classes (`Flux1FlowPipeline`, `SimformerDiffusionPipeline`,
   `Flux1JointSMPipeline`, ...) only pick a model, a `GenerativeMethod` and default params,
   and add `init_pipeline_from_config` (YAML). `ConditionalFlowPipeline`
   (`recipes/flow_pipeline.py`) is a parallel track for MAF/TarFlow: the flow *is* the density,
   so no wrapper and no `GenerativeMethod`; it trains by max-likelihood but mirrors the
   conditional surface so diagnostics run unchanged. `recipes/id_strategies.py` builds token
   id embeddings (e.g. `HealpixRope`).
5. **`inference/`** — NLE posterior sampling over trained density flows: `NLEPosterior` +
   `PosteriorTarget`, with blackjax-based `MCLMC`, `TemperedSMC`, `NestedSampler`.
6. **`diagnostics/`** — SBC, TARP, L-C2ST, marginal coverage, metrics. Works on any pipeline
   that exposes the conditional sampling surface.
7. **`experimental/`** — autoencoders + VAE pipelines, `glue` (`Embedded1DModel` /
   `Embedded2DModel` for latent-space conditioning), `FieldDiT`, `PixelDiT`,
   `FieldConditionalPipeline`. No stability promise; tests are marked `experimental`.

### Conventions that are easy to get wrong

The developer-facing ones are also written up for humans under "Development conventions" in
`CONTRIBUTING.md` (which `docs/basics/contributing.md` includes verbatim). Keep the two in sync.

- **Model call contract** (see `docs/advanced/custom_models.md`): conditional
  `model(t, obs, obs_ids, cond, cond_ids, conditioned=True)`, joint
  `model(t, obs, node_ids, condition_mask, edge_mask=None)`, unconditional
  `model(t, obs, node_ids)`. Data is always `(batch, dim, ch)`; `dim_obs` / `dim_cond` may be
  tuples for 2D data. NPE convention: `obs = theta`, `cond = x`.
- **Sampler kwargs are strict.** The method sampler/log-prob builders raise `TypeError` on
  unknown keyword arguments; model-level extras (e.g. an `encoder_key`) go through
  `model_extras={...}`, which `sample_batched` forwards per condition.
- **Mixed precision.** Master weights stay fp32; each model exposes a `dtype` knob for compute
  and emits fp32. The pipeline warns if trainable params are not fp32. Use
  `tests/precision_utils.py` (`assert_tree_dtype`, `float_leaves`) in dtype tests.
- **Sharding** (`sharding.py`): variables carry *logical* axis names via `axes()` /
  `partitioned()`; nothing is placed until `shard_model` maps them onto a mesh with a rule
  (`replicated`, `fsdp`, `tensor`). `LOGICAL_AXES`, `RULES`, `MODEL_AXIS` and the helper
  semantics are a cross-repo contract identical to `heal_swin_nnx/sharding.py`; a test pins the
  vocabulary against the installed `heal-swin-nnx`. Change both repos together.
- **Import hygiene.** `tests/test_import_smoke.py` imports `gensbi.models` and
  `gensbi.normalizing_flows` in fresh subprocesses; keep the graph acyclic (lazy imports in
  `core/__init__.py` exist for this reason).
- **Shims.** `gensbi/_compat.py` holds workarounds for third-party breakage (currently the grain
  0.2.18 absl-flag crash, applied in `gensbi/__init__.py`). Each is a no-op once upstream fixes
  ship and should then be deleted.
- **Docstrings** are NumPy style; sphinx-autoapi renders them, so public signatures and
  parameter sections matter.
- **CHANGELOG.md** follows Keep a Changelog from 0.5.0. Pre-1.0 minor releases may break; list
  such changes under **Breaking**. Update it for any user-visible change.

## Repo layout notes

- `reference/` (gitignored) holds upstream reference implementations (flowjax, ml-tarflow,
  ml-starflow) used for faithful ports; `checkpoints/` is gitignored scratch.
- `docs/superpowers/specs/` and `.superpowers/sdd/` hold design specs, task briefs and review
  diffs from past feature work; read them for intent before reworking a subsystem.
- `uv.lock` is gitignored.
