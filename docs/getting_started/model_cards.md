# Model Parameters

This page documents the main models provided in GenSBI and their configurable parameters.

- `Flux1` is the default and most powerful model to be used for most applications that involve solving inverse problems with high-dimensional data and complex posteriors. It is more memory efficient than `Simformer` and scales better to higher-dimensional data, while still being able to model complex posterior distributions.
- `Simformer` is a simpler transformer model that is easier to use for low-dimensional data and quick prototyping, which is also capable of modeling the joint distribution of all variables. If you have limited amounts of data and compute resources, and your problem is not too high-dimensional (less than ~10 dimensions) with a simple posterior, consider using `Simformer` instead of `Flux1`.
- `Flux1Joint` is a variant of `Flux1` that is specifically designed to model the joint distribution of all variables, similar to `Simformer`, but with the efficiency and scalability of `Flux1`. It performs better than `Simformer` on higher-dimensional data and complex posteriors, but performs worse for very low-dimensional data with simple posteriors (less than 4 dimensions). If your problem is likelihood dominated, and explicitly learning how to reconstruct all variables is important, consider using `Flux1Joint` instead of `Flux1`. 

---

## Simformer Model Parameters
Simformer is a transformer-based model designed to learn the joint distribution of all variables in the data, conditioned on observed subsets. It is particularly useful for low-dimensional problems and quick prototyping.

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

* **rngs**: Random number generators for model initialization, e.g. `nnx.Rngs(0)`.
* **in_channels**: Number of input channels (features) to the model. If your input data has multiple channels, set this accordingly, else set to 1.
* **dim_value**: Dimension of the value embeddings. This is the number of features used to embed the input data, the more complex the data, the higher this should be. A good starting point is around `40`.  
* **dim_id**: Dimension of the ID embeddings. This is the number of features used to embed the token id, that is the unique identifier for each token in the sequence (feature). If your data has many features, consider increasing this value, a good starting point is around `10`.
* **dim_condition**: Dimension of the condition embeddings. This is the number of features used to embed the conditioning mask, that is to say on which features the model is conditioned. A good starting point is around `10`.
* **dim_joint**: The dimension of the joint distribution to be modeled. This is the number of variables that the model will learn to represent jointly. For example, if you are modeling a 3D distribution conditioned on 2 variables, set this to `5`.
* **num_heads**: Number of attention heads in the transformer. A good starting point is `4`, and should be adjusted based on the complexity of the data and model size.
* **num_layers**: Number of transformer layers. A good starting point is `4`, and can be increased for more complex data or if the posterior distribution is expected to be complex and multimodal. 
* **num_hidden_layers**: Number of hidden layers in the transformer. This is the number of `Dense` layers per each transformer block. Default: `1`. It is rearely necessary to change this.
* **fourier_features**: Number of Fourier features for time embedding. Default: `128`. Increasing this number up to ~256 may help if the posterior distribution is expected to be multimodal.
* **widening_factor**: Widening factor for the transformer. Default: `3`. If the model is underfitting, consider increasing this value to `4`. 
* **qkv_features**: Number of features for QKV layers. Default: `None` (computed if not set). Used to bottleneck the attention mechanism to use a fixed number of features. If bottlenecking is desired, a good initial choice may be `10 * num_heads`. 

### Notes
- Currently, the Simformer model runs on `float32` precision only.
- The Simformer model is a transformer where the number of tokens is given by the number of features in the data. Each feature is treated as a token, and the model learns to represent the joint distribution of these features conditioned on some observed subset. 
- Each token is embedded to a higher-dimensional space using the `dim_value`, `dim_id`, and `dim_condition` parameters, allowing the model to capture complex relationships between features. The total number of features per token is given by `dim_tot = dim_value + dim_id + dim_condition`. As such, it is necessary to ensure that `dim_tot` is divisible by `num_heads` for the attention mechanism to work properly (else and error will be raised during model initialization).
- When choosing the model architechture, it is convenient to first increase the depth of the model (i.e. `num_layers`), and then increase the width (i.e. `dim_value` and `dim_id`) if necessary, and lastly adjust the number of attention heads. 
- If the model you would like to use has more than 8 layers, >12 heads and `dim_tot`>256, or if you would like to perform inference on more than ~10 features, consider using the **Flux1** or **Flux1Joint** models instead, as they are more memory efficient.

---

## Flux1 Model Parameters


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
    obs_dim=...,
    cond_dim=...,
    theta=...,
    guidance_embed=...,
    param_dtype=...,
)
```

**Parameter Explanations:**

* **in_channels**: Number of input channels (features) to the model.
* **vec_in_dim**: Dimension of the vector input, if applicable, default: `None`. Currently not used, and has to be set to `None`.
* **context_in_dim**: Dimension of the context input.
* **mlp_ratio**: Ratio for the MLP layers.
* **num_heads**: Number of attention heads.
* **depth**: Number of double stream blocks.
* **depth_single_blocks**: Number of single stream blocks. Should be approximately double the number of double stream blocks.
* **axes_dim**: List of dimensions for axes used in positional encoding. For 1D data, e.g. unstructured data, use a single value list, e.g. `[10]`. For higher dimensional data, it will depend on the specific positional encoding adopted, and should match the number of axes used. 
* **qkv_bias**: Whether to use bias in QKV layers. Default: `True`.
* **rngs**: Random number generators for initialization. For example `nnx.Rngs(0)`.
* **obs_dim**: Observation dimension, that is the number of variables the model needs to perform inference on.
* **cond_dim**: Condition dimension, that is the number of variables the model is conditioned on.
* **theta**: Rotary Positional Embedding (RoPE) theta parameter. A good starting point is `10 * dim_joint`, and should be tuned based on the specific data and problem. 
* **guidance_embed**: Whether to use guidance embedding. Default: `False`. Guidance embedding is currently not implemented for SBI.
* **param_dtype**: Data type for model parameters. Default: `jnp.bfloat16`. This is useful to reduce memory usage and speed up training on compatible hardware. If you experience issues with `bfloat16`, consider switching to `jnp.float32`.

---

## Flux1Joint Model Parameters


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
    joint_dim=...,
    theta=...,
    guidance_embed=...,
    param_dtype=...,
)
```

**Parameter Explanations:**

* **in_channels**: Number of input channels (features) to the model.
* **vec_in_dim**: Dimension of the vector input, if applicable.
* **mlp_ratio**: Ratio for the MLP layers.
* **num_heads**: Number of attention heads.
* **depth_single_blocks**: Number of single stream blocks.
* **axes_dim**: List of dimensions for axes used in positional encoding.
* **condition_dim**: List of dimensions for the condition used in positional encoding.
* **qkv_bias**: Whether to use bias in QKV layers.
* **rngs**: Random number generators for initialization.
* **joint_dim**: Joint dimension (number of variables modeled jointly).
* **theta**: Scaling factor for positional encoding. Default: `10_000`.
* **guidance_embed**: Whether to use guidance embedding. Default: `False`.
* **param_dtype**: Data type for model parameters. Default: `jnp.bfloat16`.

---

## Notes

- Default values may differ depending on the implementation or use case.
- For more details, see the source code in `src/gensbi/models/simformer/`, `src/gensbi/models/flux1/`, and `src/gensbi/models/flux1joint/`.

If you have further questions, please refer to the API documentation or open an issue on the repository.