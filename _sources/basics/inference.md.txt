# Inference Guide

Once your model is trained, the primary goal of Simulation-Based Inference is to generate samples from the posterior distribution $p(\theta | x)$ given a specific observation $x$.

## Basic Sampling

The `AbstractPipeline` provides a unified `sample` method for both Flow Matching and Diffusion models.

```python
import jax

# 1. Prepare your observation
# Ensure it has the shape (1, dim_cond, ch_cond)
x_observed = ... 

# 2. Generate samples
key = jax.random.PRNGKey(42)

samples = pipeline.sample(
    key=key, 
    x_o=x_observed, 
    nsamples=10_000
)

# samples shape: (10_000, dim_obs, ch_obs)
```

## Understanding Flow Matching Inference

If you are using a Flow Matching model (e.g., `Flux1FlowPipeline`), the sampling process involves solving an Ordinary Differential Equation (ODE).

1.  **Prior Sampling**: The process starts by sampling noise from a standard Normal distribution $\theta_0 \sim N(0, I)$.
2.  **ODE Integration**: The model predicts a velocity field $v_t(\theta | x)$. An ODE solver integrates this field from time $t=0$ to $t=1$ to transform the noise into samples from the posterior.

### Controlling Precision vs. Speed

The numerical integration requires discretizing the time interval $[0, 1]$. You can often control the number of steps to balance inference speed and sample quality.

```{tip}
By default, the pipeline uses a robust solver configuration (e.g., `step_size=0.01` or an adaptive solver). Reducing the number of steps by increasing the `step_size` will speed up inference but may reduce the accuracy of the posterior density.
```

## Efficient Sampling

### JIT Compilation

The `sample` method internally calls `get_sampler` to obtain a JIT-compiled sampling function, and then executes it to generate the specified number of samples. If you intend to sample multiple times separately given the same condition observation, it is recommended to call `get_sampler` directly and reuse the returned function.

```python
sampler_fn = pipeline.get_sampler(x_observed)
samples1 = sampler_fn(jax.random.PRNGKey(1), nsamples=5000)
samples2 = sampler_fn(jax.random.PRNGKey(2), nsamples=5000)
```

## Batched Inference
To perform inference efficiently on a batch of different observations (e.g., $N$ diverse inputs), use the `sample_batched` method. This handles internal batching and chunking to manage memory usage.

```python
# xs: Batch of conditions with shape (B, dim_cond, ch_cond)
xs = ... 

# Generate samples
posterior_samples = pipeline.sample_batched(
    key=jax.random.PRNGKey(42),
    condition=xs, 
    nsamples=1000,
    chunk_size=20, # Process 20 observations at a time
)

# Returns: (num_posterior_samples, B, dim_obs, ch_obs)
# e.g. (1000, B, dim_obs, 1)
```
