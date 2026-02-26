# Phase 2: Create 3 Unified Pipelines

## Goal

Add `ConditionalPipeline`, `JointPipeline`, and `UnconditionalPipeline` — each parameterized by a `GenerativeMethod` strategy. These are **model-agnostic**: any user-provided model works as long as it conforms to the wrapper interface.

Old classes stay unchanged (model-specific pipelines inherit from them). Deprecation stubs are added in Phase 4.

---

## Prerequisite: Uniform Loss Interface (`_FMLoss`)

All three strategies' `build_loss(path)` must return objects with the **same** call signature:

```python
loss_obj(model, batch, condition_mask=None, model_extras=None) → scalar
```

`_EDMLoss` and `_SMLoss` in `core/` already use this. `ContinuousFMLoss` uses `(vf, batch, **kwargs)`.

### Add `_FMLoss` wrapper

In [core/flow_matching.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/core/flow_matching.py):

1. Add `_FMLoss` class that wraps `ContinuousFMLoss` with the uniform interface
2. Change `FlowMatchingMethod.build_loss(path)` to return `_FMLoss(path)` instead of `ContinuousFMLoss(path)`

```python
class _FMLoss:
    def __init__(self, path):
        self.loss = ContinuousFMLoss(path, reduction="mean")

    def __call__(self, model, batch, condition_mask=None, model_extras=None):
        kwargs = {}
        if model_extras is not None:
            kwargs.update(model_extras)
        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask
        return self.loss(model, batch, **kwargs)
```

> [!IMPORTANT]
> Update [test_generative_method.py](file:///data/users/Aurelio/Github/GenSBI/tests/core/test_generative_method.py) `TestFlowMatchingMethod.test_build_loss` to check `isinstance(loss, _FMLoss)` instead of `ContinuousFMLoss`.

---

## File Changes

### 1. [conditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py)

**Append** `ConditionalPipeline` after the existing classes (do NOT remove old classes).

#### `__init__` signature

```python
def __init__(self, model, train_dataset, val_dataset,
             dim_obs, dim_cond, method,
             ch_obs=1, ch_cond=1,
             id_embedding_strategy=("absolute", "absolute"),
             params=None, training_config=None):
```

Key setup:
```python
self.method = method
# Merge method defaults into training config
extra = method.get_extra_training_config()
for k, v in extra.items():
    training_config.setdefault(k, v)
# Then super().__init__(...)
self.obs_ids, self.dim_obs = _resolve_embedding_ids(dim_obs, ...)
self.cond_ids, self.dim_cond = _resolve_embedding_ids(dim_cond, ...)
self.path = method.build_path(self.training_config)
self.loss_obj = method.build_loss(self.path)
```

#### `get_loss_fn` — delegates to strategy

```python
def get_loss_fn(self):
    def loss_fn(model, batch, key):
        obs, cond = batch
        prepared = self.method.prepare_batch(key, obs, self.path)
        model_extras = {"cond": cond, "obs_ids": self.obs_ids, "cond_ids": self.cond_ids}
        return self.loss_obj(model, prepared, model_extras=model_extras)
    return loss_fn
```

#### `get_sampler` — delegates to strategy

```python
def get_sampler(self, x_o, use_ema=True, **sampler_kwargs):
    model_wrapped = self.ema_model_wrapped if use_ema else self.model_wrapped
    cond = _expand_dims(x_o)
    model_extras = {"cond": cond, "obs_ids": self.obs_ids, "cond_ids": self.cond_ids}
    sampler_fn = self.method.build_sampler_fn(model_wrapped, self.path, model_extras, **sampler_kwargs)

    def sampler(key, nsamples):
        key, key_init = jax.random.split(key)
        x_init = self.method.sample_init(key_init, (nsamples, self.dim_obs, self.ch_obs), self.path)
        return sampler_fn(key, x_init)
    return sampler
```

#### Other methods

- `_wrap_model`: uses `ConditionalWrapper` (same as old classes)
- `sample(key, x_o, nsamples, ...)`: calls `get_sampler` then executes
- `init_pipeline_from_config`, `_make_model`, `get_default_params`: raise `NotImplementedError`

---

### 2. [unconditional_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/unconditional_pipeline.py)

**Append** `UnconditionalPipeline` after existing classes.

#### Key differences from Conditional

| Aspect | Conditional | Unconditional |
|---|---|---|
| `dim_cond` | user-provided | always `0` |
| `batch` from dataset | `(obs, cond)` | `obs` only (no tuple) |
| `model_extras` | `{cond, obs_ids, cond_ids}` | `{obs_ids}` |
| `get_sampler` | takes `x_o` | no `x_o` |
| `_wrap_model` | `ConditionalWrapper` | `UnconditionalWrapper` |
| `condition_mask` in loss | not used | `jnp.zeros(...)` always |

#### `__init__`

```python
def __init__(self, model, train_dataset, val_dataset,
             dim_obs, method, ch_obs=1, params=None, training_config=None):
    # super().__init__(..., dim_cond=0, ...)
    self.obs_ids, self.dim_obs = init_ids_1d(self.dim_obs)
    self.path = method.build_path(self.training_config)
    self.loss_obj = method.build_loss(self.path)
```

#### `get_loss_fn`

```python
def get_loss_fn(self):
    def loss_fn(model, batch, key):
        x_1 = batch  # NOT a tuple — just the data
        prepared = self.method.prepare_batch(key, x_1, self.path)
        condition_mask = jnp.zeros((*x_1.shape[:-1], 1), dtype=jnp.bool_)
        model_extras = {"obs_ids": self.obs_ids}
        return self.loss_obj(model, prepared, condition_mask=condition_mask, model_extras=model_extras)
    return loss_fn
```

#### `get_sampler`

```python
def get_sampler(self, use_ema=True, **sampler_kwargs):  # NO x_o
    model_wrapped = ...
    model_extras = {"obs_ids": self.obs_ids}
    sampler_fn = self.method.build_sampler_fn(model_wrapped, self.path, model_extras, **sampler_kwargs)
    def sampler(key, nsamples):
        key, key_init = jax.random.split(key)
        x_init = self.method.sample_init(key_init, (nsamples, self.dim_obs, self.ch_obs), self.path)
        return sampler_fn(key, x_init)
    return sampler
```

---

### 3. [joint_pipeline.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py)

**Append** `JointPipeline` after existing classes.

#### Key differences from Conditional

| Aspect | Conditional | Joint |
|---|---|---|
| IDs | `obs_ids`, `cond_ids` via `_resolve_embedding_ids` | `node_ids`, `obs_ids`, `cond_ids` via `init_ids_joint` |
| `batch` from dataset | `(obs, cond)` | `x_1` (concatenated obs+cond) |
| `condition_mask` | not used | sampled via `sample_condition_mask` |
| `model_extras` in loss | `{cond, obs_ids, cond_ids}` | `{node_ids}` |
| `x_init` shape in sampler | `(n, dim_obs, ch)` | `(n, dim_obs, ch)` (NOT dim_joint) |
| `model_extras` in sampler | `{cond, obs_ids, cond_ids}` | `{cond, obs_ids, cond_ids}` |
| Extra params | `id_embedding_strategy` | `condition_mask_kind` |

#### `__init__`

```python
def __init__(self, model, train_dataset, val_dataset,
             dim_obs, dim_cond, method, ch_obs=1,
             condition_mask_kind="structured",
             params=None, training_config=None):
    # super().__init__(...)
    self.dim_joint = dim_obs + dim_cond
    self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(dim_obs, dim_cond)
    self.path = method.build_path(self.training_config)
    self.loss_obj = method.build_loss(self.path)
    self.condition_mask_kind = condition_mask_kind
```

#### `get_loss_fn`

```python
def get_loss_fn(self):
    def loss_fn(model, x_1, key):
        batch_size = x_1.shape[0]
        rng_batch, rng_condition = jax.random.split(key)
        prepared = self.method.prepare_batch(rng_batch, x_1, self.path)
        condition_mask = sample_condition_mask(
            rng_condition, batch_size, self.dim_obs, self.dim_cond,
            kind=self.condition_mask_kind,
        )
        model_extras = {"node_ids": self.node_ids}
        return self.loss_obj(model, prepared, condition_mask=condition_mask, model_extras=model_extras)
    return loss_fn
```

#### `get_sampler`

```python
def get_sampler(self, x_o, use_ema=True, **sampler_kwargs):
    model_wrapped = ...
    cond = _expand_dims(x_o)
    model_extras = {"cond": cond, "obs_ids": self.obs_ids, "cond_ids": self.cond_ids}
    sampler_fn = self.method.build_sampler_fn(model_wrapped, self.path, model_extras, **sampler_kwargs)

    def sampler(key, nsamples):
        key, key_init = jax.random.split(key)
        x_init = self.method.sample_init(key_init, (nsamples, self.dim_obs, self.ch_obs), self.path)
        return sampler_fn(key, x_init)
    return sampler
```

---

### 4. [recipes/\_\_init\_\_.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/__init__.py)

Add imports and `__all__` entries for `ConditionalPipeline`, `JointPipeline`, `UnconditionalPipeline`.

---

## Tests

Write new test files:
- `tests/recipes/test_unified_conditional_pipeline.py`
- `tests/recipes/test_unified_joint_pipeline.py`
- `tests/recipes/test_unified_unconditional_pipeline.py`

Each should be parameterized over all 3 methods (`FlowMatchingMethod`, `DiffusionEDMMethod`, `ScoreMatchingMethod`) and test:
1. Pipeline init
2. Train 2 steps + sample
3. `get_loss_fn` produces scalar loss

> [!IMPORTANT]
> **All existing tests must still pass** — old classes are unchanged.

## Verification

```bash
# Existing tests (old classes)
python -m pytest tests/recipes/ tests/core/ -x --tb=short
```
