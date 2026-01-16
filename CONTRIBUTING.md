# Contributing to GenSBI

Thank you for your interest in contributing to GenSBI! We welcome contributions in the form of bug reports, feature requests, and pull requests.

## Development Setup

To set up your development environment, please follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aurelio-amerio/GenSBI.git
    cd GenSBI
    ```

2.  **Install dependencies**:
    We recommend using a virtual environment.
    ```bash
    pip install -e ".[examples,validation]"
    ```

## Running Tests

We use `pytest` for testing. To run the test suite:

```bash
pytest test/
```

Ensure all tests pass before submitting a pull request.

## Code Style

Please ensure your code adheres to the existing style conventions. We generally follow PEP 8.

## Codebase Overview

GenSBI is organized into several main components. Understanding this structure will help you navigate the codebase and make contributions effectively.

### Directory Structure

```
GenSBI/
├── src/gensbi/              # Main source code
│   ├── models/              # Neural network architectures
│   │   ├── flux1/          # Flux1 transformer model
│   │   ├── flux1joint/     # Flux1Joint variant
│   │   ├── simformer/      # Simformer model
│   │   ├── autoencoders/   # VAE and autoencoder models
│   │   ├── wrappers/       # Model wrappers for time/noise handling
│   │   └── losses/         # Loss function implementations
│   ├── recipes/             # High-level training pipelines
│   │   ├── flux1.py
│   │   ├── simformer.py
│   │   ├── base.py         # AbstractPipeline base class
│   │   └── ...
│   ├── flow_matching/       # Flow matching implementation
│   │   ├── path/           # Interpolation paths (OT, etc.)
│   │   ├── solver/         # ODE solvers
│   │   └── loss/           # Flow matching loss
│   ├── diffusion/           # Diffusion model implementation
│   │   ├── sampler/        # Diffusion samplers
│   │   ├── sde/            # Stochastic differential equations
│   │   └── loss/           # Diffusion loss functions
│   └── utils/               # Utility functions
├── test/                    # Test suite
│   ├── test_flow_matching/
│   ├── test_diffusion/
│   ├── test_models/
│   └── ...
├── docs/                    # Documentation source
│   ├── basics/             # User guides
│   ├── getting_started/    # Quick start & installation
│   ├── examples/           # Example notebooks
│   └── conf.py             # Sphinx configuration
└── pyproject.toml          # Package configuration
```

### Key Components

#### 1. Models (`src/gensbi/models/`)

Models are Flax NNX neural network modules that define the architecture:

- **Base Models**: Pure neural network definitions (e.g., `Flux1`, `Simformer`)
- **Model Wrappers**: Wrap models to provide a unified vector field and divergence interface for ODE/SDE samplers
  - `ConditionalWrapper`: For conditional inference (θ | x)
  - `JointWrapper`: For joint inference on multiple variables
  - `UnconditionalWrapper`: For unconditional density estimation

When adding a new model:
1. Create a new directory under `models/`
2. Implement the model as a `flax.nnx.Module`
3. Define a params dataclass (using `@dataclass`)
4. Add a wrapper in `models/wrappers/` if needed
5. Add tests in `test/test_models/`

#### 2. Recipes (`src/gensbi/recipes/`)

Recipes (Pipelines) are high-level interfaces that orchestrate training and inference:

- **AbstractPipeline**: Base class defining the training loop, validation, checkpointing, and EMA
- **Specific Pipelines**: Combine a model with flow matching or diffusion (e.g., `Flux1FlowPipeline`)

When adding a new pipeline:
1. Inherit from `AbstractPipeline`
2. Implement required abstract methods
3. Override `_get_optimizer()` if you need a custom optimizer
4. Override `get_default_training_config()` for custom hyperparameters

#### 3. Flow Matching (`src/gensbi/flow_matching/`)

Components for Optimal Transport Flow Matching:

- **Paths** (`path/`): Define interpolation between source and target (e.g., `CondOTProbPath`)
- **Solvers** (`solver/`): ODE solvers for inference (e.g., `ODESolver`)
- **Loss** (`loss/`): Flow matching loss functions

#### 4. Diffusion (`src/gensbi/diffusion/`)

Components for diffusion models:

- **SDEs** (`sde/`): Define noise schedules (VP, VE, EDM)
- **Samplers** (`sampler/`): Inference samplers (e.g., `EulerSampler`)
- **Loss** (`loss/`): Diffusion loss functions (score matching, denoising)

### How to Add New Features

#### Adding a New Model Architecture

1. Create `src/gensbi/models/your_model/`
2. Implement the model:
   ```python
   from flax import nnx
   from dataclasses import dataclass
   
   @dataclass
   class YourModelParams:
       num_layers: int
       hidden_dim: int
       # ... other params
   
   class YourModel(nnx.Module):
       def __init__(self, params: YourModelParams):
           # Initialize layers
           pass
       
       def __call__(self, x, context=None):
           # Forward pass
           return output
   ```
3. Add a pipeline recipe in `src/gensbi/recipes/your_model.py`
4. Write tests in `test/test_models/test_your_model.py`
5. Add a Model Card in `docs/basics/model_cards.md`

#### Adding a New Training Feature

1. If it's pipeline-specific: Override methods in your pipeline class
2. If it's general: Add to `AbstractPipeline` in `src/gensbi/recipes/base.py`
3. Add configuration options to `get_default_training_config()`
4. Write tests in `test/test_recipes/`

#### Adding a New Loss Function

1. Create the loss in `src/gensbi/models/losses/` or `src/gensbi/flow_matching/loss/`
2. Make it compatible with the pipeline's `get_loss_fn()` method
3. Add tests in the appropriate test directory

### Testing Guidelines

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Ensure tests are deterministic (use fixed random seeds)
- Run the full test suite before submitting a PR

Example test structure:
```python
import pytest
from gensbi.models.your_model import YourModel, YourModelParams

def test_your_model_forward_pass():
    params = YourModelParams(num_layers=2, hidden_dim=64)
    model = YourModel(params)
    # ... test logic
```

### Documentation Guidelines

When adding new features:

1. **Docstrings**: Add Google-style docstrings to all public functions/classes
   ```python
   def your_function(arg1: int, arg2: str) -> float:
       """Brief description.
       
       Longer description if needed.
       
       Args:
           arg1: Description of arg1.
           arg2: Description of arg2.
           
       Returns:
           Description of return value.
       """
   ```

2. **User Guides**: Update relevant guides in `docs/basics/` if your feature affects users

3. **Model Cards**: If adding a new model, add a section to `docs/basics/model_cards.md`

4. **Examples**: Consider adding a simple example in the docstring or a notebook in the examples repo

### Pull Request Process

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Implement your feature with tests
3. **Run tests**: `pytest test/`
4. **Update docs**: Add/update documentation as needed
5. **Commit**: Use clear, descriptive commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Open PR**: Provide a clear description of changes and motivation

### Code Review

All PRs will be reviewed for:
- Correctness and functionality
- Test coverage
- Code style and clarity
- Documentation completeness
- Compatibility with existing features

## Questions?

If you have questions about contributing:
- Check the [Conceptual Overview](/basics/overview) to understand the architecture
- Read the [API Documentation](/api/gensbi/index)
- Open an issue on GitHub for discussion

We appreciate your contributions and look forward to collaborating with you!
