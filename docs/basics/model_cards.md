# Model Cards

This page documents the neural network architectures provided in GenSBI. These models serve as the core generative engines for approximating posterior distributions in Simulation-Based Inference (SBI).

Selecting the appropriate model is crucial for balancing computational efficiency with the ability to capture complex, high-dimensional dependencies. The models below are designed to cover a wide range of use cases, from rapid prototyping on low-dimensional problems to solving large-scale inverse problems.

## Quick Model Comparison

| Model | Best For | Dimensions | Memory | Strengths | Limitations |
|-------|----------|------------|--------|-----------|-------------|
| **Flux1** | Most applications | High (>10) | Excellent | Scalable, memory-efficient, RoPE embeddings | Only for conditional models |
| **Simformer** | Rapid prototyping | Low (<10) | Good | Explicit embeddings, simple, fast for low-dim | Poor scaling to high-dim |
| **Flux1Joint** | Joint modeling | Medium-High | Good | Explicit joint learning, scalable | Slightly less complex than Flux1 (no double stream layers)|

### When to Use Each Model

- **Flux1** (Default): Use for most problems, especially when:
  - You have >10 parameters or >100 observations
  - Memory efficiency is important
  - You need scalability to high dimensions

- **Simformer**: Use when:
  - You have <10 total dimensions
  - You want rapid prototyping on simple problems
  - You prefer explicit variable ID embeddings

- **Flux1Joint**: Use when:
  - You need explicit joint modeling of all variables
  - Your problem is likelihood-dominated
  - You have medium to high dimensional problems (4-100 dimensions)

## Model Descriptions

- **Flux1**: The robust default choice for most applications. It excels at solving inverse problems involving high-dimensional data and complex posterior distributions. Unlike `Simformer`, `Flux1` embeds only the data explicitly and relies on Rotary Positional Embeddings (RoPE) for variable identification. This approach is significantly more memory-efficient and scales better to higher dimensions.
- **Simformer**: A lightweight transformer model optimized for low-dimensional data and rapid prototyping. It explicitly models the joint distribution of all variables by embedding values, variable IDs, and condition masks separately. This explicit embedding strategy is highly effective for low-dimensional data (fewer than ~10 dimensions) as it compresses the data less than RoPE, but it is less computationally efficient for high-dimensional problems.
- **Flux1Joint**: Combines the joint-distribution modeling capabilities of `Simformer` with the scalable architecture of `Flux1`. It adopts the `Flux1` embedding strategy (explicit data embedding + RoPE for IDs), making it ideal for high-dimensional problems where explicitly learning the joint reconstruction of variables is crucial. While it outperforms `Simformer` on complex, high-dimensional tasks, `Simformer` is often preferable for very low-dimensional problems (less than 4 dimensions) due to its superior explicit ID embedding.

## ID Embedding Strategies

Transformers process data as sequences of tokens. To differentiate between unordered parameters (sets) and structured data (grids/sequences), GenSBI uses various ID embedding strategies.

**For a detailed guide on ID strategies (Absolute vs. RoPE) and data preprocessing, please see [Data, IDs, and Embeddings](/basics/data_and_embeddings).**

### Strategy Support by Model

| Model | Obs & Cond Separation | Supported Strategies | Default |
| :--- | :--- | :--- | :--- |
| **Flux1** | **Separate**<br>Distinct embeddings for parameters (obs) and data (cond). | `absolute`, `pos1d`/`rope1d`, `pos2d`/`rope2d` | `absolute` (Params), `absolute` (Data) |
| **Flux1Joint** | **Unified**<br>All variables are part of a single joint sequence. | `absolute`, `pos1d`/`rope1d`, `pos2d`/`rope2d` | `absolute` |
| **Simformer** | **Unified**<br>Explicit learned embeddings for all tokens. | `absolute` (learned) | `absolute`<br>*(Note: Embeddings are **concatenated** to features, not summed)* |

```{note}
`Flux1` allows mixing strategies. For example, you can use `absolute` embeddings for your unordered physical parameters ($\theta$) while using `pos1d` or `rope` for your sequential observational data ($x$).
```

```{warning}
**Preferred Embedding Strategies**:
While the codebase supports the generic `rope` keyword (which adapts to N-dimensions), it is **strongly recommended** to use `rope1d` for sequential data and `rope2d` for image/grid data. This ensures compatibility with the helper functions in the pipelines.
```

```{warning}
While `Flux1Joint` technically supports `rope`, it is generally recommended to use `absolute` embeddings for joint density estimation of unordered variables to avoid imposing artificial relational biases.
```

## Flux1 Model Parameters

Flux1 is a scalable architecture using double-stream blocks, capable of handling high-dimensional inputs efficiently.

**How to use:**

```python
from gensbi.models.flux1 import Flux1Params

params = Flux1Params(
    in_channels=...,
    vec_in_dim=None,
    context_in_dim=...,
    mlp_ratio=...,
    num_heads=...,
    depth=...,
    depth_single_blocks=...,
    axes_dim=...,
    qkv_bias=...,
    rngs=...,
    dim_obs=...,
    dim_cond=...,
    theta=...,
    id_embedding_strategy=("absolute", "absolute"),
    guidance_embed=...,
    param_dtype=...,
)
```

**Parameter Explanations:**

- **in_channels**: Number of input channels in the data (e.g., `1` for scalar/vector fields, `3` for images). This is distinct from the number of features or tokens.
- **vec_in_dim**: Dimension of the vector input (e.g., time embeddings). **Must be set to `None`** as it is currently unused.
- **context_in_dim**: Dimension of the context (conditioning) input (similar to in_channels)
- **mlp_ratio**: The expansion ratio for the MLP layers within transformer blocks (typically `4.0`).
- **num_heads**: Number of attention heads.
- **depth**: Number of Double Stream blocks (processes information and context separately).
- **depth_single_blocks**: Number of Single Stream blocks (processes information and context jointly). A common heuristic is to set this to roughly double the `depth`.
- **axes_dim**: A sequence of integers defining the number of features per attention head, per axis. For 1D data, this is a single-element list defining the per-head dimension. The total number of transformer features is `sum(axes_dim) * num_heads`. For unstructured 1D data, a typical value is around `[10]` or greater.
- **qkv_bias**: Whether to use bias terms in QKV projections. Default: `True`.
- **rngs**: Random number generators for initialization (e.g., `nnx.Rngs(0)`).
- **dim_obs**: The number of variables (tokens) the model performs inference on.
- **dim_cond**: The number of variables the model is conditioned on.
- **theta**: Scaling factor for Rotary Positional Embeddings (RoPE). A recommended starting point is `10 * dim_obs`. The default code value is `10_000`.
- **id_embedding_strategy**: A tuple of strings `(obs_kind, cond_kind)` specifying the embedding strategy for observation and condition tokens respectively. Options: `"absolute"`, `"pos1d"`, `"pos2d"`, `"rope"`. Default: `("absolute", "absolute")`.
- **guidance_embed**: Whether to use guidance embeddings. Default: `False` (not currently implemented for SBI).
- **param_dtype**: Data type for model parameters. Default: `jnp.bfloat16`. Use this to reduce memory usage. Switch to `jnp.float32` if you encounter numerical stability issues.

### Notes on Flux1

- **Architecture Configuration**: It is strongly recommended to use double the number of Single Stream blocks (`depth_single_blocks`) compared to the number of Double Stream blocks (`depth`).
- **Tuning Strategy**: A typical depth range for the model is between 8 and 20. For the attention mechanism, starting with 6-8 heads and approximately 10 features per head is recommended; these can be increased based on data complexity.
- **High-Dimensional Data**: If your condition dimension is large (>100) or observation dimension is moderately high (>20), it is highly recommended to employ an embedding network to derive summary statistics for the data. See the latent diffusion example (WIP).

## Simformer Model Parameters

Simformer is a transformer-based model designed to learn the joint distribution of all variables in the data, conditioned on observed subsets. It treats features as tokens, allowing it to capture complex dependencies in low-dimensional spaces.

**How to use:**

```python
from gensbi.models.simformer import SimformerParams

params = SimformerParams(
    rngs=...,
    in_channels=...,
    dim_value=...,
    dim_id=...,
    dim_condition=...,
    dim_joint=...,
    num_heads=...,
    num_layers=...,
    num_hidden_layers=...,
    fourier_features=...,
    widening_factor=...,
    qkv_features=...,
)
```

**Parameter Explanations:**

- **rngs**: Random number generators for model initialization (e.g., `nnx.Rngs(0)`).
- **in_channels**: Number of input channels in the data (e.g., `1` for scalar/vector fields). This defines the depth of the input tensor, not the number of features or tokens.
- **dim_value**: The dimension of the value embeddings. This determines the size of the feature representation inside the model. Higher values allow modeling more complex data; a good starting point is `40`.
- **dim_id**: The dimension of the ID embeddings. This embeds the unique identifier for each variable (token). For datasets with many variables, consider increasing this; a good starting point is `10`.
- **dim_condition**: The dimension of the condition embeddings. This represents the conditioning mask (i.e., which variables are observed vs. unobserved). A good starting point is `10`.
- **dim_joint**: The total number of variables to be modeled jointly (the sequence length). For example, modeling a 3D distribution conditioned on 2 observed variables would require a `dim_joint` of 5.
- **num_heads**: Number of attention heads. A standard starting point is `4`. Adjust based on data complexity and model size constraints.
- **num_layers**: Number of transformer layers. A default of `4` works well for many problems. Increase this for complex, multimodal posterior distributions.
- **num_hidden_layers**: Number of dense hidden layers within each transformer block. Default: `1`. It is rarely necessary to change this.
- **fourier_features**: Number of Fourier features used for time embeddings. Default: `128`. Increasing this to ~256 may help resolve multimodal posteriors.
- **widening_factor**: The expansion factor for the internal feed-forward layers. Default: `3`. If the model is underfitting, try increasing to `4`.
- **qkv_features**: Dimension of the Query/Key/Value projection. Default: `None` (automatically computed). Setting this allows you to bottleneck the attention mechanism. A manual setting might be `10 * num_heads`.

### Notes on Simformer

- **Precision**: Currently, the Simformer model runs on `float32` precision only.
- **Architecture**: The model treats every variable in the data as a distinct token. It learns the joint distribution of these tokens conditioned on an observed subset.
- **Embedding Dimensions**: The total embedding size for a token is `dim_tot = dim_value + dim_id + dim_condition`. This sum **must** be divisible by `num_heads` to ensure correct attention splitting; otherwise, initialization will fail.
- **Tuning Strategy**: Start by increasing `num_layers` (depth). If performance is still lacking, increase `dim_value` and `dim_id` (width), and finally adjust `num_heads`.
- **Limitations**: If your problem requires more than 8 layers, >12 heads, `dim_tot > 256`, or inference on >10 variables, `Flux1` or `Flux1Joint` are recommended for better memory efficiency.

## Flux1Joint Model Parameters

Flux1Joint utilizes a pure Single Stream architecture (similar to Simformer but using Flux layers) to model the joint distribution of variables efficiently.

**How to use:**

```python
from gensbi.models.flux1joint import Flux1JointParams

params = Flux1JointParams(
    in_channels=...,
    vec_in_dim=...,
    mlp_ratio=...,
    num_heads=...,
    depth_single_blocks=...,
    axes_dim=...,
    condition_dim=...,
    qkv_bias=...,
    rngs=...,
    dim_joint=...,
    theta=...,
    id_embedding_strategy="absolute",
    guidance_embed=...,
    param_dtype=...,
)
```

**Parameter Explanations:**

- **in_channels**: Number of input channels in the data (e.g., `1` for scalar/vector fields). This is distinct from the number of features or tokens.
- **vec_in_dim**: Dimension of the vector input, typically used for timestep embeddings.
- **mlp_ratio**: The expansion ratio for the MLP layers within the transformer blocks (typically `4.0`).
- **num_heads**: Number of attention heads. Ensure `in_channels` is divisible by this number.
- **depth_single_blocks**: The total number of transformer layers. Since `Flux1Joint` relies entirely on Single Stream blocks to mix joint information, this defines the total depth of the network.
- **axes_dim**: A sequence of integers defining the number of features per attention head for the **joint variables** (the target variables being modeled). For 1D unstructured data, a typical value is around `[10]` or greater.
- **condition_dim**: A list with the number of features to be used to encode the condition mask in each token. Should match in dimension with `axes_dim`.
- **qkv_bias**: Whether to use bias terms in QKV projections. Default: `True`.
- **rngs**: Random number generators for initialization (e.g., `nnx.Rngs(0)`).
- **dim_joint**: The number of variables to be modeled jointly. This equates to the sequence length of the target tokens.
- **theta**: Scaling factor for Rotary Positional Embeddings (RoPE). Default: `10_000`.
- **id_embedding_strategy**: String specifying the embedding strategy (e.g., `"absolute"`, `"rope"`). Default: `"absolute"`.
- **guidance_embed**: Whether to use guidance embeddings. Default: `False`.
- **param_dtype**: Data type for model parameters. Default: `jnp.bfloat16`.

### Notes on Flux1Joint

- **When to use**: If your problem is likelihood dominated, and explicitly learning how to reconstruct all variables is important, consider using `Flux1Joint` instead of `Flux1`.
- **Performance Comparison**: `Flux1Joint` typically outperforms `Simformer` on higher-dimensional data and complex posteriors. However, it may perform worse for very low-dimensional data with simple posteriors (less than 4 dimensions).
- **Tuning Strategy**: A typical depth range for the model is between 8 and 20. For the attention mechanism, starting with 6-8 heads and approximately 10 features per head is recommended; these can be increased based on data complexity.
- **High-Dimensional Data**: If your condition dimension is large (>100) or observation dimension is moderately high (>20), it is highly recommended to employ an embedding network to derive summary statistics for the data. See the latent diffusion example (WIP).

## Notes

- **Default Values**: Specific default values may vary based on the exact version of the library. Always check the function signatures if unsure.
- **Source Code**: For deeper implementation details, refer to:
  - `src/gensbi/models/simformer/`
  - `src/gensbi/models/flux1/`
  - `src/gensbi/models/flux1joint/`

If you have further questions, please refer to the API documentation or open an issue on the repository.