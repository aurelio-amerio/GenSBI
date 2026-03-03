# Custom Models

GenSBI's unified pipelines (`ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline`) are **model-agnostic** — you can use any Flax NNX model, not just the built-in Flux1 and Simformer architectures. This page explains how to write a custom model that works with the pipelines.

## How the Pipeline Calls Your Model

During training, the pipeline calls your model directly. During sampling, it wraps your model in a `ModelWrapper` that provides a standard interface for ODE/SDE solvers.

The wrapper calls your model's `__call__` method with specific arguments depending on the **pipeline type**:

### Conditional Pipeline

```python
# The wrapper calls your model as:
model(t, obs, obs_ids, cond, cond_ids, conditioned=True)
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `t` | `(B,)` or `(B, 1)` | Time steps |
| `obs` | `(B, dim_obs, ch_obs)` | Noisy observations (parameters being sampled) |
| `obs_ids` | `(B, dim_obs, ...)` | ID embeddings for observation tokens |
| `cond` | `(B, dim_cond, ch_cond)` | Conditioning data (fixed during sampling) |
| `cond_ids` | `(B, dim_cond, ...)` | ID embeddings for conditioning tokens |
| `conditioned` | `bool` | Always `True` during sampling |

### Joint Pipeline

```python
# The wrapper calls your model as:
model(t, obs, node_ids, condition_mask, edge_mask=None)
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `t` | `(B,)` or `(B, 1)` | Time steps |
| `obs` | `(B, dim_joint, ch_obs)` | Joint data (parameters + conditioning concatenated) |
| `node_ids` | `(B, dim_joint)` | Token identifiers |
| `condition_mask` | `(dim_joint,)` | Boolean mask: `True` for conditioned tokens |
| `edge_mask` | Optional | Attention mask (if provided) |

### Unconditional Pipeline

```python
# The wrapper calls your model as:
model(t, obs, node_ids)
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `t` | `(B,)` or `(B, 1)` | Time steps |
| `obs` | `(B, dim_obs, ch_obs)` | Data being modeled |
| `node_ids` | `(B, dim_obs, ...)` | Token identifiers |

---

## Writing a Custom Model

### For Transformer-Based Models

If your model is a transformer that uses token IDs for positional information, implement the exact signature shown above for your pipeline type. The built-in `Flux1` and `Simformer` models are good references:

- [Flux1](../api/gensbi/models/flux1/model/index) — for `ConditionalPipeline`
- [Simformer](../api/gensbi/models/simformer/model/index) — for `JointPipeline`

### For Non-Transformer Models (MLPs, CNNs, etc.)

If your custom model is **not** a transformer, it won't use `node_ids`, `obs_ids`, `cond_ids`, or `condition_mask`. Use `*args` and `**kwargs` to absorb these unused arguments:

```python
from flax import nnx
import jax
import jax.numpy as jnp
from gensbi.utils.model_wrapping import _expand_dims, _expand_time


class CustomMLP(nnx.Module):
    """Example custom model for unconditional estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        din = input_dim + 1  # +1 for time

        self.layers = [
            nnx.Linear(din, hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, input_dim, rngs=rngs),
        ]

    def __call__(self, t, obs, *args, **kwargs):
        # *args absorbs node_ids, condition_mask, etc.
        # **kwargs absorbs edge_mask, conditioned, etc.

        obs = _expand_dims(obs)[..., 0]  # (B, dim, ch) -> (B, dim)
        t = _expand_time(t)
        t = jnp.broadcast_to(t, (obs.shape[0], 1))

        h = jnp.concatenate([obs, t], axis=-1)
        for layer in self.layers[:-1]:
            h = jax.nn.gelu(layer(h))
        h = self.layers[-1](h)

        return h[..., None]  # (B, dim) -> (B, dim, 1)
```

```{important}
The key rule: your model's `__call__` must accept `(t, obs, ...)` as its first two positional arguments and return a tensor with the **same shape** as `obs`. Use `*args` and `**kwargs` to absorb any additional arguments that the wrapper passes but your model doesn't need.
```

### For Conditional Non-Transformer Models

For a conditional MLP, you need to handle `cond` as well:

```python
class ConditionalMLP(nnx.Module):
    def __init__(self, dim_obs, dim_cond, hidden_dim=128, *, rngs: nnx.Rngs):
        din = dim_obs + dim_cond + 1  # obs + cond + time
        self.linear1 = nnx.Linear(din, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, dim_obs, rngs=rngs)

    def __call__(self, t, obs, obs_ids, cond, cond_ids, *args, **kwargs):
        # obs_ids and cond_ids are unused but must be accepted
        obs_flat = _expand_dims(obs)[..., 0]
        cond_flat = _expand_dims(cond)[..., 0]
        t = _expand_time(t)
        t = jnp.broadcast_to(t, (obs_flat.shape[0], 1))

        h = jnp.concatenate([obs_flat, cond_flat, t], axis=-1)
        h = jax.nn.gelu(self.linear1(h))
        h = jax.nn.gelu(self.linear2(h))
        h = self.linear3(h)

        return h[..., None]  # same shape as obs
```

---

## Using Custom Models with Unified Pipelines

Once your model follows the interface, pass it directly to the pipeline:

```python
from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod

model = ConditionalMLP(dim_obs=3, dim_cond=5, rngs=nnx.Rngs(42))

pipeline = ConditionalPipeline(
    model,
    train_dataset,
    val_dataset,
    dim_obs=3,
    dim_cond=5,
    method=FlowMatchingMethod(),
)

pipeline.train(rngs=nnx.Rngs(0))
samples = pipeline.sample(key, x_o, nsamples=10_000)
```

For available generative methods and solvers, see the [Samplers and Solvers](samplers.md) documentation.

---

## Real-World Examples

For examples of custom models used in scientific applications, see:

- **Gravitational lensing**: [lensing_example.ipynb](../notebooks/lensing_example.ipynb) — custom CNN encoder with conditional pipeline
- **Gravitational waves**: [gw_example.ipynb](../notebooks/gw_example.ipynb) — custom architecture for GW parameter estimation

```{note}
These notebooks are being updated to use the new unified pipeline interface. The model interface patterns shown above apply.
```

---

## Working Examples

See these example scripts that demonstrate custom and built-in models with unified pipelines:

- [unconditional_pipeline.py](https://github.com/aurelio-amerio/GenSBI/blob/main/docs/examples/unconditional_pipeline.py) — Custom MLP + `UnconditionalPipeline`
- [conditional_pipeline.py](https://github.com/aurelio-amerio/GenSBI/blob/main/docs/examples/conditional_pipeline.py) — Flux1 + `ConditionalPipeline`
- [joint_pipeline.py](https://github.com/aurelio-amerio/GenSBI/blob/main/docs/examples/joint_pipeline.py) — Simformer + `JointPipeline`
