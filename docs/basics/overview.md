# Conceptual Overview: How GenSBI is Structured

This page explains the core concepts and architecture of GenSBI to help you understand how the different components work together.

## High-Level Architecture

GenSBI is organized around three main abstractions:

```{mermaid}
graph TB
    subgraph Pipeline["Pipeline<br/>(High-level orchestration: training, validation, etc.)"]
        subgraph Model["Model<br/>(Neural network architecture: Flux1, etc.)"]
            Wrapper["Model Wrapper<br/>(Adds time/noise handling logic)"]
        end
        FlowDiff["Flow/Diffusion<br/>(Loss function, ODE solver, paths)"]
    end
    
    style Pipeline fill:#e1f5ff,stroke:#333,stroke-width:2px
    style Model fill:#fff4e6,stroke:#333,stroke-width:2px
    style Wrapper fill:#f0f0f0,stroke:#333,stroke-width:1px
    style FlowDiff fill:#fff4e6,stroke:#333,stroke-width:2px
```

## Core Concepts

### 1. Models

**Models** are the neural network architectures that learn to approximate posterior distributions. They are standard Flax NNX modules.

GenSBI provides three main model architectures:

- **Flux1**: A double-stream transformer using Rotary Position Embeddings (RoPE). Best for high-dimensional problems.
- **Simformer**: A single-stream transformer that explicitly embeds variable IDs. Best for low-dimensional problems.
- **Flux1Joint**: A single-stream variant of Flux1 for explicit joint modeling. Good for likelihood-dominated problems.

**Example:**
```python
from gensbi.models.flux1 import Flux1, Flux1Params
from flax import nnx

params = Flux1Params(
    in_channels=1,
    num_heads=8,
    depth=12,
    depth_single_blocks=24,
    axes_dim=[obs_dim],
    rngs=nnx.Rngs(0),
    obs_dim=3,
    cond_dim=5,
)

model = Flux1(params)
```

### 2. Model Wrappers

**Model Wrappers** provide a standard interface for models to be used by ODE/SDE solvers during sampling. They standardize how models are called and provide methods for computing the vector field and divergence needed for numerical integration.

Three types of wrappers exist:

- **Unconditional**: For unconditional density estimation
- **Conditional**: For conditional inference (standard SBI: estimate θ given x)
- **Joint**: For joint inference (estimate multiple variables simultaneously)

The wrapper provides:
- Standardized calling interface for solvers
- `get_vector_field()` method for ODE integration
- `get_divergence()` method when needed for likelihood computation

**Note**: Wrappers are only used during sampling/inference. During training, the unwrapped model is called directly.

### 3. Recipes and Pipelines

**Recipes** define complete end-to-end procedures for a specific task (e.g., SBI, VAE training). **Pipelines** are specific implementations of these recipes using particular generative modeling approaches (e.g., flow matching or diffusion).

Currently, GenSBI provides two main recipes:
- **SBI Recipe**: For simulation-based inference
- **VAE Recipe**: For training variational autoencoders

**Pipelines** handle all aspects of training and inference:

- Data loading and batching
- Training loop (optimizer, learning rate scheduling, early stopping)
- Validation and checkpointing
- Exponential Moving Average (EMA) of weights
- Model wrapping for sampling

**Key SBI Pipelines:**
- `Flux1FlowPipeline`: Flow matching with Flux1 model
- `SimformerFlowPipeline`: Flow matching with Simformer model
- `Flux1JointFlowPipeline`: Flow matching with Flux1Joint model
- Similar diffusion variants exist

**Example:**
```python
from gensbi.recipes import Flux1FlowPipeline

pipeline = Flux1FlowPipeline(
    train_dataset=train_iter,
    val_dataset=val_iter,
    obs_dim=3,
    cond_dim=5,
    params=flux1_params,
)

# Train
pipeline.train(rngs=nnx.Rngs(0))

# Sample (note: x_o is the observed data, not 'condition')
samples = pipeline.sample(rng=key, x_o=x_observed, nsamples=10_000)
```

### 4. Flow Matching vs. Diffusion

GenSBI supports two approaches for generative modeling:

#### Flow Matching (Recommended)
- **Concept**: Learn a velocity field that transports samples from a simple distribution (Gaussian noise) to the target distribution (posterior).
- **Training**: The model learns to predict velocity at random time points. The model directly defines a vector field as a function of (obs, cond, t).
- **Sampling**: Solve an ODE from t=0 to t=1 using the learned velocity field.
- **Advantages**: Straighter paths in latent space, faster sampling, easier to train.
- **Sampling**: Solve an ODE from t=0 to t=1.
- **Advantages**: Straighter paths in latent space, faster sampling, easier to train.

#### Diffusion
- **Concept**: Learn to gradually denoise data that has been corrupted with noise.
- **Training**: Predict the noise or score at different noise levels.
- **Sampling**: Iteratively denoise starting from pure noise.
- **Note**: As of the current version, flow matching models tend to be more stable and easier to train than diffusion models. This may change in future releases.

**Flow Matching is the recommended default in GenSBI.**

## How Components Work Together

Here's what happens during training:

1. **Data Loading**: The pipeline gets batches of (observations, conditions) from your dataset.

2. **Loss Computation**:
   - Sample random time steps `t ∈ [0, 1]`
   - Create noisy versions of the data based on `t`
   - The model (unwrapped) predicts the velocity/noise as a function of (obs, cond, t)
   - Compare prediction to ground truth

3. **Optimization**:
   - Compute gradients
   - Update model parameters
   - Update EMA shadow weights

4. **Validation**:
   - Periodically evaluate on validation set
   - Save checkpoints if performance improves
   - Early stopping if validation loss diverges

During inference:

1. **ODE Solving** (Flow Matching):
   - Wrap the model to provide standard interface for the solver
   - Start with Gaussian noise
   - Use the wrapped model's `get_vector_field()` method with an ODE solver
   - Result: samples from the posterior distribution

2. **Iterative Denoising** (Diffusion):
   - Wrap the model for the SDE sampler
   - Start with pure noise
   - Iteratively denoise using the learned denoiser
   - Result: samples from the posterior distribution

## File Organization

The codebase is organized into logical modules:

```
src/gensbi/
├── models/              # Neural network architectures
│   ├── flux1/          # Flux1 model
│   ├── flux1joint/     # Flux1Joint model
│   ├── simformer/      # Simformer model
│   ├── wrappers/       # Time/noise handling wrappers
│   └── losses/         # Loss functions
├── recipes/             # High-level training pipelines
│   ├── flux1.py
│   ├── simformer.py
│   └── ...
├── flow_matching/       # Flow matching components
│   ├── path/           # Interpolation paths
│   ├── solver/         # ODE solvers
│   └── loss/           # Flow matching loss
├── diffusion/           # Diffusion components
│   ├── sampler/        # Diffusion samplers
│   ├── sde/            # SDE definitions
│   └── loss/           # Diffusion loss
└── utils/               # Utility functions
```

## Design Principles

GenSBI follows these design principles:

1. **Modularity**: Components (models, wrappers, losses, solvers) are independent and composable.

2. **Sensible Defaults**: Pipelines come with reasonable default hyperparameters that work for many problems.

3. **Easy Customization**: You can override specific methods (e.g., optimizer, loss function) without rewriting everything.

4. **JAX-Native**: Built on JAX and Flax NNX for performance, automatic differentiation, and hardware acceleration.

5. **SBI-Focused**: Designed specifically for simulation-based inference, not general-purpose generative modeling.

## What's a "Recipe"?

The term **recipe** comes from the idea of providing a pre-packaged, tested combination of components that work well together—like a cooking recipe. Instead of manually combining a model, wrapper, loss, optimizer, and training loop, a recipe gives you a one-line solution:

```python
pipeline = Flux1FlowPipeline(train_data, val_data, obs_dim, cond_dim, params)
pipeline.train(rngs)
samples = pipeline.sample(key, x_observed)
```

Behind the scenes, the recipe handles all the complexity.

## Next Steps

Now that you understand the structure:

1. **Choose a Model**: See [Model Cards](/basics/model_cards) for guidance.
2. **Set Up Training**: Follow the [Training Guide](/basics/training).
3. **Run Inference**: See the [Inference Guide](/basics/inference).
4. **Validate Results**: Use the [Validation Guide](/basics/validation).
5. **Try Examples**: Explore the [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples).

If you want to extend GenSBI or add custom components, see the [Contributing Guide](/CONTRIBUTING) and the [API Documentation](/api/gensbi/index).
