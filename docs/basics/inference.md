# Inference Guide

Once your model is trained, the primary goal of Simulation-Based Inference is to generate samples from the posterior distribution $p(\theta | x)$ given a specific observation $x$.

## Basic Sampling

The `AbstractPipeline` provides a unified `sample` method for both Flow Matching and Diffusion models.

```python
import jax

# 1. Prepare your observation
# Ensure it has the shape (1, cond_dim, cond_channels)
x_observed = ... 

# 2. Generate samples
key = jax.random.PRNGKey(42)

samples = pipeline.sample(
    key, 
    condition=x_observed, 
    num_samples=10_000
)

# samples shape: (10_000, obs_dim, obs_channels)
```

## Understanding Flow Matching Inference

If you are using a Flow Matching model (e.g., `Flux1FlowPipeline`), the sampling process involves solving an Ordinary Differential Equation (ODE).

1.  **Prior Sampling**: The process starts by sampling noise from a standard Normal distribution $\theta_0 \sim N(0, I)$.
2.  **ODE Integration**: The model predicts a velocity field $v_t(\theta | x)$. An ODE solver integrates this field from time $t=0$ to $t=1$ to transform the noise into samples from the posterior.

### Controlling Precision vs. Speed

The numerical integration requires discretizing the time interval $[0, 1]$. You can often control the number of steps to balance inference speed and sample quality.

> [!NOTE]
> By default, the pipeline uses a robust solver configuration (e.g., `dt=0.01` or an adaptive solver). Reducing the number of steps will speed up inference but may reduce the accuracy of the posterior density.

## Efficient Sampling

### JIT Compilation

The `sample` method internally calls `get_sampler` to obtain a JIT-compiled sampling function, and then executes it to generate the specified number of samples. If you intend to sample multiple times separately given the same condition observation, it is recommended to call `get_sampler` directly and reuse the returned function.

```python
sampler_fn = pipeline.get_sampler(x_observed)
samples1 = sampler_fn(jax.random.PRNGKey(1), num_samples=5000)
samples2 = sampler_fn(jax.random.PRNGKey(2), num_samples=5000)
```

<!-- ### Batching over Observations

If you need to perform inference for a batch of different observations $x_1, x_2, ..., x_N$, you should use `jax.vmap` over the pipeline's sampler or loop efficiently.

```python
# Example of vmapping the sampler
def get_samples(key, x):
    return pipeline.sample(key, condition=x, num_samples=1000)

keys = jax.random.split(key, num_observations)
batch_samples = jax.vmap(get_samples)(keys, batch_of_observations)
``` -->

<!-- ## Post-Processing

The samples returned are JAX arrays. For analysis, you typically want to convert them to NumPy arrays or use them directly in validation metrics.

```python
import numpy as np
samples_np = np.array(samples)
``` -->
