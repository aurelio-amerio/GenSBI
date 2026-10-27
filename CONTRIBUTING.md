# Contributing to GenSBI

Thank you for your interest in contributing to GenSBI! All contributions are welcome, whether it's improving the documentation, implementing new models for flow matching or score matching, fixing bugs, or anything else.

Feel free to [open an issue](https://github.com/aurelio-amerio/GenSBI/issues) to propose new features, report bugs, or discuss ideas.

## Development Setup

Python 3.12 or higher is required. We recommend [uv](https://docs.astral.sh/uv/).

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aurelio-amerio/GenSBI.git
    cd GenSBI
    ```

2.  **Install dependencies**:
    ```bash
    # editable install plus the dev dependency group (lint, test, docs, notebooks)
    uv sync --group dev

    # same, with CUDA 13 JAX and the example extras
    uv sync --group cuda13full

    # or using pip (>= 25.1, which understands dependency groups)
    pip install -e ".[examples]" --group dev
    ```

## Requirements for Contributions

- **Tests**: All new functionality must include comprehensive tests. We use `pytest`; run the full suite with `uv run pytest` and ensure all tests pass before submitting a PR. The next section lists the markers CI uses.
- **Documentation**: All public functions and classes must be documented using **NumPy-style docstrings**. The API reference is generated from them with sphinx-autoapi.
- **Code style**: Please follow PEP 8 conventions. CI fails on syntax errors and undefined names (`flake8 --select=E9,F63,F7,F82`) and reports everything else at a 127-character line length as warnings.
- **Changelog**: Add an entry to `CHANGELOG.md` for every user-visible change. The file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); while GenSBI is pre-1.0, minor releases may break the API, and such changes must be listed under a **Breaking** heading.

## Running the tests

`pyproject.toml` configures pytest to run on CPU (`JAX_PLATFORMS=cpu`) with two xdist workers, eight fake host devices for the sharding tests, and XLA flags that make floating-point results identical across CI runners. Do not override these when reproducing a CI failure.

```bash
uv run pytest                                                        # whole suite
uv run pytest -m "not slow and not experimental and not extraslow"   # CI "fast" tier
uv run pytest -m slow                                                # CI "slow" tier
uv run pytest -m experimental                                        # CI "experimental" tier
uv run pytest tests/core/test_ode_solver.py::test_name -n 0          # one test, without xdist
```

Tests marked `extraslow` are end-to-end recovery runs and never execute in CI; run them locally when touching the code they cover.

## Development conventions

These are the rules that are easy to break without noticing, because the thing that enforces them lives in a different file from the one you are editing.

- **JAX floor.** CI runs the fast tier a second time with `jax==0.11.1` and `jaxlib==0.11.1` pinned. New code must work at the minimum JAX version declared in `pyproject.toml`, not only at the latest release. If you need a newer JAX feature, raise the floor in `pyproject.toml` deliberately and note it in the changelog.

- **Import graph stays acyclic.** `tests/test_import_smoke.py` imports `gensbi.models` and `gensbi.normalizing_flows` in fresh subprocesses, so a circular import that happens to work in the test process still fails CI. `gensbi.core` lazy-loads its method classes for this reason. If a new module needs something from a higher layer, prefer a lazy import inside the function over a module-level one.

- **Sharding vocabulary is a cross-repo contract.** `LOGICAL_AXES`, `RULES`, `MODEL_AXIS` and the semantics of the helpers in `src/gensbi/sharding.py` must stay identical to `heal_swin_nnx/sharding.py` in the [heal-swin-nnx](https://pypi.org/project/heal-swin-nnx/) package. Neither package imports the other; a test in `tests/sharding/` pins the vocabulary against the installed `heal-swin-nnx`. Change both repositories together.

- **Notebooks under `docs/` are downloaded copies.** `make download` in `docs/` wipes `docs/notebooks/` and refetches every notebook from the [GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) repository, along with `getting_started/my_first_model.ipynb` and `advanced/methods_and_samplers.ipynb`. Fix notebooks there, not here; edits made in this repository are lost on the next download.

- **Third-party workarounds live in `src/gensbi/_compat.py`.** Each shim documents the upstream bug it works around and is written to be a no-op once the fix ships, so it can be deleted when the corresponding dependency floor is raised past it.

- **Custom models follow a fixed call signature.** The pipelines call models positionally; see the [Custom Models](https://gensbi.com/advanced/custom_models.html) page before adding an architecture.

## Directory Structure

```
GenSBI/
├── src/gensbi/              # Main source code
│   ├── core/                # GenerativeMethod strategies (flow matching, EDM, score matching)
│   ├── models/              # Neural network architectures
│   │   ├── core/            # Shared patching, tokenizers, stats
│   │   ├── embedding/       # Embedding layers
│   │   ├── flux1/           # Flux1 transformer model
│   │   ├── flux1joint/      # Flux1Joint variant
│   │   ├── simformer/       # Simformer model
│   │   ├── maf/             # Masked autoregressive flow
│   │   ├── tarflow/         # Transformer autoregressive flow
│   │   ├── losses/          # Loss function implementations
│   │   └── wrappers/        # Model wrappers for ODE/SDE interface
│   ├── recipes/             # High-level training pipelines
│   │   ├── pipeline.py      # AbstractPipeline base class
│   │   ├── conditional_pipeline.py
│   │   ├── unconditional_pipeline.py
│   │   ├── joint_pipeline.py
│   │   ├── flow_pipeline.py # Discrete normalizing-flow (NPE) pipeline
│   │   ├── flux1.py
│   │   ├── flux1joint.py
│   │   ├── simformer.py
│   │   └── id_strategies.py # Token-id embedding strategies
│   ├── flow_matching/       # Flow matching implementation
│   │   ├── path/            # Interpolation paths
│   │   ├── solver/          # ODE solvers
│   │   ├── loss/            # Flow matching loss
│   │   └── utils/           # Flow matching utilities
│   ├── diffusion/           # Diffusion model implementation
│   │   ├── path/            # Noise schedules (VP, VE, EDM)
│   │   └── solver/          # Diffusion samplers
│   ├── normalizing_flows/   # Bijections and change-of-variables primitives
│   ├── inference/           # NLE posterior sampling (MCLMC, tempered SMC, nested sampling)
│   ├── diagnostics/         # Diagnostic and validation tools
│   │   ├── metrics/         # Evaluation metrics
│   │   ├── lc2st.py         # LC2ST diagnostic
│   │   ├── sbc.py           # Simulation-Based Calibration
│   │   ├── tarp.py          # TARP diagnostic
│   │   └── marginal_coverage.py
│   ├── experimental/        # Experimental features
│   │   ├── models/
│   │   └── recipes/
│   ├── sharding.py          # Multi-device mesh helpers
│   ├── _compat.py           # Shims for third-party breakage
│   └── utils/               # Utility functions
├── tests/                   # Test suite, mirrors src/gensbi
├── docs/                    # Documentation source
│   ├── basics/              # User guides
│   ├── getting_started/     # Installation & quick start
│   ├── advanced/            # Custom models, samplers, normalizing flows
│   ├── theoretical_overview/# Theory behind flow & diffusion models
│   ├── examples/            # Example pipeline scripts and YAML configs
│   ├── notebooks/           # Tutorial notebooks (downloaded, see above)
│   └── conf.py              # Sphinx configuration
├── CHANGELOG.md
└── pyproject.toml           # Package configuration
```

## Pull Request Process

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with tests, documentation and a changelog entry
3. Run tests: `uv run pytest`
4. Push and open a PR with a clear description of your changes

We appreciate your contributions and look forward to collaborating with you!
