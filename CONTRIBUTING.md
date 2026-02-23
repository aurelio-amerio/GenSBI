# Contributing to GenSBI

Thank you for your interest in contributing to GenSBI! All contributions are welcome, whether it's improving the documentation, implementing new models for flow matching or score matching, fixing bugs, or anything else.

Feel free to [open an issue](https://github.com/aurelio-amerio/GenSBI/issues) to propose new features, report bugs, or discuss ideas.

## Development Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aurelio-amerio/GenSBI.git
    cd GenSBI
    ```

2.  **Install dependencies** (we recommend using a virtual environment):
    ```bash
    pip install -e ".[cuda12,examples,test,docs]"
    ```

## Requirements for Contributions

- **Tests**: All new functionality must include comprehensive tests. We use `pytest`; run the full suite with `pytest test/` and ensure all tests pass before submitting a PR.
- **Documentation**: All public functions and classes must be documented using **NumPy-style docstrings**.
- **Code style**: Please follow PEP 8 conventions.

## Directory Structure

```
GenSBI/
├── src/gensbi/              # Main source code
│   ├── models/              # Neural network architectures
│   │   ├── embedding/       # Embedding layers
│   │   ├── flux1/           # Flux1 transformer model
│   │   ├── flux1joint/      # Flux1Joint variant
│   │   ├── simformer/       # Simformer model
│   │   ├── losses/          # Loss function implementations
│   │   └── wrappers/        # Model wrappers for ODE/SDE interface
│   ├── recipes/             # High-level training pipelines
│   │   ├── pipeline.py      # AbstractPipeline base class
│   │   ├── conditional_pipeline.py
│   │   ├── unconditional_pipeline.py
│   │   ├── joint_pipeline.py
│   │   ├── flux1.py
│   │   ├── flux1joint.py
│   │   ├── simformer.py
│   │   └── config_examples/ # Example configuration files
│   ├── flow_matching/       # Flow matching implementation
│   │   ├── path/            # Interpolation paths
│   │   ├── solver/          # ODE solvers
│   │   ├── loss/            # Flow matching loss
│   │   └── utils/           # Flow matching utilities
│   ├── diffusion/           # Diffusion model implementation
│   │   ├── path/            # Noise schedules (VP, VE, EDM)
│   │   └── solver/          # Diffusion samplers
│   ├── diagnostics/         # Diagnostic and validation tools
│   │   ├── metrics/         # Evaluation metrics
│   │   ├── lc2st.py         # LC2ST diagnostic
│   │   ├── sbc.py           # Simulation-Based Calibration
│   │   ├── tarp.py          # TARP diagnostic
│   │   └── marginal_coverage.py
│   ├── experimental/        # Experimental features
│   │   ├── models/
│   │   └── recipes/
│   └── utils/               # Utility functions
├── test/                    # Test suite
│   ├── diagnostics/
│   ├── diffusion/
│   ├── experimental/
│   ├── flow_matching/
│   ├── models/
│   ├── recipes/
│   └── utils/
├── docs/                    # Documentation source
│   ├── basics/              # User guides
│   ├── getting_started/     # Installation & quick start
│   ├── theoretical_overview/# Theory behind flow & diffusion models
│   ├── examples/            # Example pipeline scripts
│   ├── notebooks/           # Tutorial notebooks
│   └── conf.py              # Sphinx configuration
└── pyproject.toml           # Package configuration
```

## Pull Request Process

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with tests and documentation
3. Run tests: `pytest test/`
4. Push and open a PR with a clear description of your changes

We appreciate your contributions and look forward to collaborating with you!
