# Troubleshooting & FAQ

This page addresses common issues and frequently asked questions when using GenSBI.

## Installation Issues

### CUDA/GPU Not Detected

**Problem**: JAX is not detecting your GPU, or you're getting CUDA-related errors.

**Solution**:
1. Ensure you installed the correct JAX version for your CUDA version. **CUDA 12 is recommended**:
   ```bash
   pip install "GenSBI[cuda12] @ git+https://github.com/aurelio-amerio/GenSBI.git"
   ```
   Note: CUDA 11 is not officially supported. CUDA 13 support will be available in an upcoming release.

2. Verify JAX can see your GPU:
   ```python
   import jax
   print(jax.devices())  # Should show GPU devices
   ```

3. If issues persist, check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### Import Errors

**Problem**: Getting `ModuleNotFoundError` or import errors.

**Solution**:
1. Ensure GenSBI is installed correctly:
   ```bash
   pip install git+https://github.com/aurelio-amerio/GenSBI.git
   ```

2. Check your Python version (requires Python 3.11+).

## Training Issues

### Shape Mismatch Errors

**Problem**: Getting errors like "incompatible shapes" or dimension mismatches.

**Solution**:
1. **Check data shapes**: GenSBI expects data in the format `(batch, features, channels)`.
   - For scalar features: `(batch, num_features, 1)`
   - Example: 3 parameters → shape `(batch_size, 3, 1)`

2. **Verify dim_obs and dim_cond**: These should match the number of features (not including channels).
   ```python
   # If theta has shape (batch, 3, 1) and x has shape (batch, 5, 1)
   dim_obs = 3   # Number of parameters
   dim_cond = 5  # Number of observations
   ```

3. **Check what is a token/dimension and what is a channel: for 1D unstructured data, set channel=1**.

#### Meaning of `dim_obs` / `dim_cond` and `ch_obs` / `ch_cond`

Shape bugs often come from mixing up **how many observables you have** with **how many values each observable carries**.

GenSBI represents both “parameters to infer” ($\theta$) and “conditioning data” ($x$) as 3D tensors:

- Parameters (a.k.a. *obs* in the pipeline API): `theta` has shape `(batch, dim_obs, ch_obs)`.
- Conditioning data (a.k.a. *cond*): `x` has shape `(batch, dim_cond, ch_cond)`.

Different parts of the library/docs may use different names for the same concepts:

- `dim_obs`: number of *parameter tokens* (how many parameters you infer).
- `dim_cond`: number of *conditioning tokens* (how many observables are measured / provided to the model).
- `ch_obs`: number of channels per parameter token.
- `ch_cond`: number of channels per conditioning token.

**Rule of thumb**:

- `*_dim` answers: “How many distinct observables/tokens do I have?”
- `*_channels` / `ch_*` answers: “How many values/features does each observable/token carry?”

Most SBI problems use **one channel for parameters** (`ch_obs = 1`), because you typically want **one token per parameter**.

Conditioning data often has **more than one channel** (`ch_cond >= 1`), because each measured “token” may carry multiple features.

##### Concrete example: 2 GW parameters, 2 detectors, frequency grid

Suppose your simulator parameters are two scalars $\theta = (\theta_1, \theta_2)$, and your observation is a frequency-domain strain measured by **two detectors** on the same frequency grid with `n_lambda` frequency bins.

- Parameters tensor (`theta`):
   - `dim_obs = 2` (two parameters)
   - `ch_obs = 1` (each parameter is a scalar)
   - shape: `(batch, 2, 1)`

- Conditioning tensor (`x`):
   - `dim_cond = n_lambda` (one token per frequency bin)
   - `ch_cond = 2` (two detector strain values per frequency)
   - shape: `(batch, n_lambda, 2)`

In other words: the *frequency grid lives in* `dim_cond`, while the *detector index lives in* `ch_cond`.

If later you decide to store more features per frequency bin (e.g., real/imag parts, or multiple summary statistics per detector), you typically increase `ch_cond` while keeping `dim_cond = n_lambda`.

### Training Loss Not Decreasing

**Problem**: Loss stays flat or doesn't improve during training.

**Solution**:
1. **Increase batch size**: Flow matching and diffusion models benefit from large batch sizes (ideally 1024+) to cover the time interval well. If your GPU memory is limited, use gradient accumulation (`multistep`) to achieve a large effective batch size (e.g., physical batch of 128 × multistep of 8 = 1024 effective batch size).

2. **Check learning rate**: Default is `1e-3`. Try reducing to `1e-4` or increasing to `5e-4`.

3. **Verify data**: Ensure your simulator is producing valid, varied samples.

4. **Model size**: Your model might be too small. Try increasing `depth`, `num_heads`, or feature dimensions.

### Training Diverges or NaN Loss

**Problem**: Loss becomes NaN or explodes during training.

**Solution**:
1. **Check data normalization**: Extreme values can cause instability. Consider normalizing your data to a reasonable range (e.g., [-1, 1] or [0, 1]). Ideally, normalize both data and parameters to have zero mean and unit variance for best results.

2. **Reduce learning rate**: Try `max_lr=1e-4` or lower.

3. **Use float32 precision**: If using `bfloat16`, switch to `float32` in model parameters:
   ```python
   params = Flux1Params(..., param_dtype=jnp.float32)
   ```

4. **Gradient clipping**: Although not in default config, you may need to add gradient clipping to your custom optimizer.

5. **Check `theta` for RoPE**: If using `rope1d` or `rope2d` embeddings and the model goes to NaN after a few epochs, the base frequency `theta` might be wrong.
   - **Rule of thumb**: Use `theta = 10 * dim_rope_dimensions`.
   - **Example (1D)**: Imagine `obs` uses absolute ID embedding and `cond` uses `rope1d`. If `cond` has 7 tokens, `theta` should be `~10 * 7 = 70` (usually rounded up to 100).
   - **Example (2D)**: If `cond` uses `rope2d` with 32x32 images as input (patch size 2x2), the number of patches is 16x16=256. `theta` should be `~16 * 16 * 10 = 2560`.
   - If both `obs` and `cond` use RoPE, sum the recommended results for each.
   - **Note**: If the image is larger than 32x32, it is strongly advisable to first encode it using a CNN (see the gravitational lensing example).

### Memory Errors (OOM)

**Problem**: GPU runs out of memory during training.

**Solution**:
1. **Reduce batch size**: Lower your DataLoader batch size.

2. **Use gradient accumulation**: Set `training_config["multistep"]` to accumulate gradients over multiple steps:
   ```python
   training_config["multistep"] = 4  # Effective batch = batch_size * 4
   ```

3. **Use bfloat16**: Switch model parameters to `param_dtype=jnp.bfloat16` (default for Flux1).

4. **Reduce model size**: Decrease `depth`, `depth_single_blocks`, or `num_heads`.

5. **Use a smaller model**: Consider using `Simformer` for low-dimensional problems instead of `Flux1`.

## Multiprocessing Issues

### Multiprocessing Hangs or Crashes

**Problem**: Script hangs when using multiprocessing with grain or similar data loaders.

**Solution**:
1. **Guard GPU initialization**: Add this at the very top of your script:
   ```python
   import os
   if __name__ != "__main__":
       os.environ["JAX_PLATFORMS"] = "cpu"
   else:
       os.environ["JAX_PLATFORMS"] = "cuda"
   ```

2. **Use `if __name__ == "__main__":`**: Wrap your main code in this guard:
   ```python
   if __name__ == "__main__":
       main()
   ```

3. See the [Training Guide](/basics/training) for a complete multiprocessing example.

## Inference Issues

### Samples Don't Look Right

**Problem**: Posterior samples are unrealistic or don't match expectations.

**Solution**:
1. **Use EMA model**: Ensure you're using the EMA version of your model (loaded from `checkpoints/ema/`).

2. **Increase sampling steps**: If using a custom ODE solver, increase the number of integration steps.

3. **Check conditioning**: Verify that `x_observed` has the correct shape `(1, dim_cond, ch_cond)` and values.

4. **Run validation diagnostics**: Use SBC, TARP, or L-C2ST to check if your model is well-calibrated. See the [Validation Guide](/basics/validation).

### Slow Inference

**Problem**: Sampling takes too long.

**Solution**:
1. **Use JIT compilation**: Call `get_sampler()` once and reuse the function:
   ```python
   sampler_fn = pipeline.get_sampler(x_observed)
   samples = sampler_fn(jax.random.PRNGKey(42), num_samples=10_000)
   ```

2. **Batch sampling**: Generate samples in batches rather than one at a time.

3. **Consider Flow Matching**: Flow matching learns straighter trajectories than diffusion, often allowing for fewer integration steps (faster sampling) without sacrificing quality.

## Validation Issues

### SBC/TARP/L-C2ST Errors

**Problem**: Errors when running validation diagnostics from the `gensbi.diagnostics` module.

**Solution**:
1. **Check array shapes**: Diagnostics expect flattened 2D arrays `(num_samples, features)`. GenSBI data usually comes in 3D `(batch, features, channels)`.
   ```python
   # GenSBI format: (batch, features, channels)
   # Diagnostics format: (batch, features * channels)
   thetas_flat = thetas.reshape(thetas.shape[0], -1)
   ```

2. **Check data types**: Ensure you are passing Numpy or JAX arrays, not PyTorch tensors.

3. **Use separate validation data**: Don't use training data for validation diagnostics.

4. See the [Validation Guide](/basics/validation) for detailed examples.

## Model Selection

### Which Model Should I Use?

**Question**: Should I use Flux1, Simformer, or Flux1Joint?

**Answer**:
- **Flux1** (default): Best for most applications, especially high-dimensional problems (>10 parameters or >100 observations). Very memory efficient.
- **Simformer**: Best for low-dimensional problems (<10 parameters total) and rapid prototyping. Easiest to understand.
- **Flux1Joint**: Best when you need explicit joint modeling of all variables. Often better for likelihood-dominated problems. Falls between Flux1 and Simformer in memory efficiency.

See [Model Cards](/basics/model_cards) for detailed comparisons.

### How Many Layers/Heads Should I Use?

**Question**: How do I choose the right model size?

**Answer**:
**Starting points:**
- **Flux1**: `depth=4-8`, `depth_single_blocks=8-16`, `num_heads=6-8`
- **Simformer**: `num_layers=4-6`, `num_heads=4-6`, `dim_value=40`

**Tuning strategy:**
1. Start with default/recommended values
2. If underfitting, increase depth first (number of layers)
3. Then increase width (heads, feature dimensions)
4. Monitor memory usage and training time

## Data Preparation

### How Should I Structure My Data?

**Question**: What format should my training data be in?

**Answer**:
The data format depends on whether you're using a conditional or joint estimator:

**Conditional methods (e.g., Flux1)**: Expect tuples `(obs, cond)` where:
- `obs`: parameters to infer, shape `(batch, dim_obs, ch_obs)`
- `cond`: conditioning data (observations), shape `(batch, dim_cond, ch_cond)`

**Joint estimators (e.g., Flux1Joint, Simformer)**: Expect a single "joint" sample of shape `(batch, dim_joint, channels)`.

**Important**: Joint estimators only work well when both obs and cond share the same data structure. If your observations are fundamental parameters but your conditioning data is a time series or 2D image, use a conditional density estimator instead, as it will perform better by preserving the structure of the data rather than treating everything as a joint distribution.

For scalar data, `channels = 1`.

Example:
```python
def split_obs_cond(data):
    # data shape: (batch, dim_obs + dim_cond, 1)
    return data[:, :dim_obs], data[:, dim_obs:]

train_dataset = (
    grain.MapDataset.source(data)
    .shuffle(seed=42)
    .repeat()  # Infinite iterator
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
)
```

## Getting More Help

If your issue isn't covered here:

1. **Check the Examples**: The [GenSBI-examples repository](https://github.com/aurelio-amerio/GenSBI-examples) contains working examples.

2. **Read the Guides**: See [Training](/basics/training), [Inference](/basics/inference), [Validation](/basics/validation), and [Model Cards](/basics/model_cards).

3. **Open an Issue**: Report bugs or ask questions on the [GitHub Issues page](https://github.com/aurelio-amerio/GenSBI/issues).

4. **API Documentation**: Check the [API Reference](/api/gensbi/index) for detailed function signatures.
