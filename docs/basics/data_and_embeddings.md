# Data, IDs, and Embeddings

GenSBI uses a flexible system to handle various data modalities, including unstructured parameter sets, time-series, and 2D images. This page explains how data is structured, how to preprocess it, and how the model identifies different variables using ID embeddings.

## Data Structure: Tokens and Channels

GenSBI models (like Flux1) operate on tensors with the shape `(batch, num_tokens, num_channels)`.

*   **Tokens (`dim`):** Represent distinct variables or observables.
    *   For **unstructured data** (e.g., physical parameters $\theta_1, \theta_2$), each parameter is a token.
    *   For **time-series**, each time step is a token.
    *   For **images**, the image is broken into patches, and each patch is a token.
*   **Channels (`ch`):** Represent the features per token.
    *   For simple parameters, this is usually 1.
    *   For a gravitational wave detector reading at a specific time, this could be 1 (amplitude).
    *   For an image patch, this could be the number of pixels in that patch (e.g., $patch\_size \times patch\_size \times color\_channels$).

Distinguishing between *what* a token is (its ID) and *what value* it holds is key to the model's performance.

## Preprocessing

Proper data preprocessing is critical for efficient training.

### Normalization

To speed up convergence, **ensure your data is normalized**.
*   **Standardization**: Shift and scale your data so that it has approximately 0 mean and unit variance.
    *   Apply this to both parameters (inference targets) and observations (conditioning data).

### Patchification (for Images)

Transformers process sequences, not grids. If you are using 2D image data, you **must patchify** it before passing it to the model (or pipeline).

The function `gensbi.recipes.utils.patchify_2d` flattens a 2D image into a sequence of tokens.

```python
from gensbi.recipes.utils import patchify_2d

# Input shape: (Batch, Height, Width, Channels)
# e.g., (32, 64, 64, 3)
images = ... 

# Patchify
# Output shape: (Batch, Num_Tokens, Features_Per_Token)
# With 2x2 patches: (32, 32*32, 2*2*3) = (32, 1024, 12)
tokens = patchify_2d(images)
```

## ID Embeddings

Since Transformers are permutation-invariant by default, we use **ID Embeddings** to tell the model the identity or position of each token. This applies to both the **variables you want to infer (observations/parameters)** and the **data you condition on (conditions)**.

### Strategies

GenSBI supports several embedding strategies via the `id_embedding_strategy` parameter (a tuple for `(obs, cond)`):

1.  **`absolute` (Learned)**
    *   **Use for:** Unstructured data where order doesn't matter (e.g., a set of independent cosmological parameters).
    *   **Mechanism:** The model learns a unique vector for each token index.
    *   **Initialization:** Use `init_ids_1d`.

2.  **`pos1d` / `rope1d` (1D Positional)**
    *   **Use for:** Sequential data (e.g., time series, spectra).
    *   **Mechanism:** Encodes the 1D index ($t=1, t=2, \dots$). `rope1d` uses Rotary Positional Embeddings, which are generally superior for capturing relative distances.
    *   **Initialization:** Use `init_ids_1d`.

3.  **`pos2d` / `rope2d` (2D Positional)**
    *   **Use for:** Image data or 2D grids.
    *   **Mechanism:** Encodes the 2D coordinates $(x, y)$ of the token. `rope2d` extends RoPE to two dimensions.
    *   **Initialization:** Use `init_ids_2d`.

### Initialization Example

When using one of the default pipelines, like the `ConditionalFlowPipeline`, **ID initialization is handled automatically** based on your `dims` and `id_embedding_strategy`.

However, if you are using the models directly or need custom handling, here is how to initialize the IDs for both observations and conditions.

```python
import jax.numpy as jnp
from gensbi.recipes.utils import init_ids_1d, init_ids_2d

# Example: 
# Obs: 5 unstructured parameters (absolute)
# Cond: 64x64 image (rope2d)

dim_obs = 5
dim_cond = (64, 64) # passed as tuple implies 2D

# --- Observation IDs (Unstructured) ---
# semantic_id=0 identifies these as "observation" tokens
obs_ids = init_ids_1d(dim_obs, semantic_id=0)
# Shape: (1, 5, 2) -> (Batch per device, Num_Tokens, ID_Features)

# --- Condition IDs (Image) ---
# semantic_id=1 identifies these as "condition" tokens
cond_ids = init_ids_2d(dim_cond, semantic_id=1)
# Shape: (1, 1024, 3) -> (Batch per device, Num_Tokens, ID_Features) 
# Note: 64x64 image -> 32x32 patches = 1024 tokens

print(f"Obs IDs shape: {obs_ids.shape}")
print(f"Cond IDs shape: {cond_ids.shape}")
```

### Automatic Pipeline Handling

If you use the recipes (e.g., `ConditionalFlowPipeline`), you simply specify the structure:

```python
pipeline = ConditionalFlowPipeline(
    model=...,
    dim_obs=5,                     # 5 tokens
    dim_cond=(64, 64),             # Image dimensions
    id_embedding_strategy=("absolute", "rope2d"), # Obs=Absolute, Cond=RoPE 2D
    ...
)
```

The pipeline will automatically detecting that `dim_cond` is a tuple and use `init_ids_2d` (and expects you to pass patchified data during training).

## Working with 2D Images & Spatial Data

GenSBI provides first-class support for 2D data (like images from telescopes or simulations), but it requires specific preprocessing.

### 1. Patchification is Mandatory

The models process data as a sequence of tokens. A standard 2D image must be broken down into patches.

Use `gensbi.recipes.utils.patchify_2d` to convert your image tensors `(Batch, H, W, C)` into token sequences `(Batch, Num_Tokens, Features)`.

**Example Workflow:**

1.  **Load Image Data**: Shape `(N, 64, 64, 3)`
2.  **Patchify**:
    ```python
    x_patch = patchify_2d(images, patch_size=2)
    # New Shape: (N, 1024, 12)
    # 32*32 patches = 1024 tokens
    # 2*2*3 pixels per patch = 12 channels per token
    ```
3.  **Feed to Pipeline**: Pass `x_patch` as your conditioning data.

### 2. Use `rope2d` for Spatial Awareness

When your data is an image, the *2D spatial relationship* between patches is crucial.

*   **Avoid** `absolute` or `rope1d`: These treat the image as a long 1D line, losing the knowledge that pixel (0,0) is close to (0,1) AND (1,0).
*   **Use** `rope2d`: This embedding strategy encodes the grid structure. The model will understand the 2D distance between patches.

To use this in the `Flux1` model:

```python
params = Flux1Params(
    ...
    dim_cond=1024,                  # Total number of patches
    id_embedding_strategy=("absolute", "rope2d"), # Obs=Params, Cond=Image
)
```

**Note:** When manually creating IDs with `init_ids_2d`, pass the *grid dimensions* (e.g., `(32, 32)`), not the total number of tokens.

### 3. Shape Mismatches

Common error: Passing the raw image `(N, 64, 64, 3)` directly to the model.

*   **Symptom**: Shape errors complaining about rank or dimension mismatches.
*   **Fix**: Ensure you call `patchify_2d` before training and before inference.

