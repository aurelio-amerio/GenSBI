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

**Model Wrappers** add the logic for handling time steps, noise, and conditioning to the base neural network. They bridge the gap between the model architecture and the training algorithm (flow matching or diffusion).

Three types of wrappers exist:

- **Unconditional**: For unconditional density estimation
- **Conditional**: For conditional inference (standard SBI: estimate θ given x)
- **Joint**: For joint inference (estimate multiple variables simultaneously)

The wrapper handles:
- Time embedding (converting time `t ∈ [0, 1]` to a format the model can use)
- Noise/signal combination at different time steps
- Conditioning information formatting

**Example:**
```python
from gensbi.models.wrappers import ConditionalWrapper

wrapped_model = ConditionalWrapper(model)
# Now the model can be called with time, data, and conditions
output = wrapped_model(time, noisy_data, condition_data)
```

### 3. Recipes (Pipelines)

**Recipes** (also called **Pipelines**) are high-level interfaces that combine everything needed for training and inference. They handle:

- Data loading and batching
- Training loop (optimizer, learning rate scheduling, early stopping)
- Validation and checkpointing
- Exponential Moving Average (EMA) of weights
- Sampling from the trained model

**Key Pipelines:**
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

# Sample
samples = pipeline.sample(key, condition=x_observed, num_samples=10_000)
```

### 4. Flow Matching vs. Diffusion

GenSBI supports two approaches for generative modeling:

#### Flow Matching (Recommended)
- **Concept**: Learn a velocity field that transports samples from a simple distribution (Gaussian noise) to the target distribution (posterior).
- **Training**: Minimize the difference between predicted and true velocity at random time points.
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
   - The wrapped model predicts the velocity/noise
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
   - Start with Gaussian noise
   - Use the learned velocity field to solve an ODE
   - Result: samples from the posterior distribution

2. **Iterative Denoising** (Diffusion):
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
