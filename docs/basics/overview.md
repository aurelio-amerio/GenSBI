# Conceptual Overview: How GenSBI is Structured

This page explains the core concepts and architecture of GenSBI to help you understand how the different components work together.

## High-Level Architecture

GenSBI is built upon three core abstractions:

- Models: Neural architectures such as Flux1 and Simformer.
- Sampling Algorithms: Primarily Flow Matching and Diffusion. Each abstraction defines its own ODE/SDE formulations and implements the corresponding solvers.
- Pipelines: Workflows that orchestrate the end-to-end process of training, validation, and sampling.

```{image} ../_static/pipeline_graph.png
:alt: GenSBI pipeline
:align: center
:width: 400px
```
<br>
Changing or customizing any of these components allows you to adapt GenSBI to your specific inference problems.

## Core Concepts

### 1. Models

**Models** are the neural network architectures that learn to approximate posterior distributions. They are standard Flax NNX modules.

GenSBI provides three main model architectures:

- **Flux1**: A double-stream transformer using Rotary Position Embeddings (RoPE). Best for high-dimensional problems.
- **Simformer**: A single-stream transformer that explicitly embeds variable IDs. Best for low-dimensional problems.
- **Flux1Joint**: A single-stream variant of Flux1 for explicit joint modeling. Good for likelihood-dominated problems.

**For detailed comparisons and selection guides, see [Model Cards](/basics/model_cards).**

```{note}
GenSBI represents both parameters ($\theta$) and observations ($x$) with the tensor convention `(batch, dim, channels)`.

- `dim_obs`: number of parameter tokens (how many parameters you infer).
- `dim_cond`: number of conditioning tokens (how many observables you provide to the model).
- `ch_obs` and `ch_cond`: number of values carried by each token.

Most SBI problems use `ch_obs = 1` (one scalar per parameter token), while `ch_cond` can be > 1 (e.g., multiple detectors or multiple features per measurement). See [Troubleshooting: Shape Mismatch Errors](/basics/troubleshooting#shape-mismatch-errors) for a concrete example.
```

### 2. Model Wrappers

**Model Wrappers** provide a standard interface for models to be used by ODE/SDE solvers during sampling. They standardize how models are called and provide methods for computing the vector field and divergence needed for numerical integration.

Three types of wrappers exist:

- **Unconditional**: For unconditional density estimation
- **Conditional**: For conditional inference (standard SBI: estimate θ given x)
- **Joint**: For joint inference (estimate multiple variables simultaneously)

The wrapper provides:
- Standardized calling interface for solvers
- `get_vector_field()` method for ODE/SDE solution (used for Flow and Diffusion models)
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
    dim_obs=3,
    dim_cond=5,
    params=flux1_params,
)

# Train
pipeline.train(rngs=nnx.Rngs(0))

# Sample from posterior p(theta|x_o)
# x_o is the observed measurement data used to condition the density estimation
samples = pipeline.sample(key=key, x_o=x_observed, nsamples=10_000)
```

### 4. Flow Matching vs. Diffusion

GenSBI supports two approaches for generative modeling:

#### Flow Matching (Recommended)
- **Concept**: Learn a time-dependent **vector field** $v_t(x)$ that transports samples from a simple prior (Gaussian noise) to the target data distribution.
- **Training**: The model directly regresses the vector field generating the probability paths. We typically use Optimal Transport paths (straight lines), which leads to a vector field that is stable and easy to learn.
- **Sampling**: Solve an Ordinary Differential Equation (ODE) from t=0 to t=1 along the learned vector field. The linear trajectories allow for fast and robust integration.
- **Why it's better**: Flow matching generally offers **faster training** and **faster sampling** than diffusion. The vector field with Optimal Transport paths behaves better than the score function, leading to straighter sampling trajectories that require fewer steps to solve.

#### Diffusion
- **Concept**: Learn to reverse a stochastic process that gradually adds noise to the data.
- **Training**: The model learns the **score function** $\nabla \log p_t(x)$ (or equivalently predicts the noise) at different noise levels to reverse the corruption process.
- **Sampling**: Solve a Stochastic Differential Equation (SDE) or ODE to iteratively denoise the samples.
- **Pros/Cons**: Diffusion models can sometimes offer greater **sample diversity** due to the stochastic nature of SDEs. However, the score function can be harder to learn (singularities at small noise), and the non-linear reverse paths typically require more sampling steps.

**Flow Matching is the recommended default in GenSBI.**

**For a deeper mathematical dive, see the [Theoretical Overview](/theoretical_overview/index).**

## How Components Work Together

Here's what happens during training:

1. **Data Loading**: The pipeline gets batches of (observations, conditions) from your dataset.

2. **Loss Computation**:
   - Sample random time steps `t ∈ [0, 1]`
   - Create noisy versions of the data based on `t`
   - The model predicts the velocity/noise as a function of (obs, cond, t)
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
   - Start with pure noise (sampled according to the SDE prior distribution)
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

5. **Density Estimation Focus**: Designed for conditional and unconditional density estimation with applications in simulation-based inference (neural posterior estimation, neural likelihood estimation, neural prior estimation) and general conditional density estimation tasks.

## What's a "Recipe"?

The term **recipe** comes from the idea of providing a pre-packaged, tested combination of components that work well together—like a cooking recipe. Instead of manually combining a model, wrapper, loss, optimizer, and training loop, a recipe gives you a one-line solution:

```python
pipeline = Flux1FlowPipeline(train_data, val_data, dim_obs, dim_cond, params)
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
5. **Try Examples**: Explore the [Examples](/examples) page and the [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples).

If you want to extend GenSBI or add custom components, see the [Contributing Guide](/CONTRIBUTING) and the [API Documentation](/api/gensbi/index).
