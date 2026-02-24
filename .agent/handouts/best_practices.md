# Best Practices for GenSBI Development

## Environment Setup

### Activating the correct environment

`mamba run -n gensbi` can silently use the wrong Python environment. **Always** activate explicitly:

```bash
mamba deactivate && mamba deactivate && mamba activate gensbi
```

Deactivating twice ensures you escape any nested environments. Only then run commands directly:

```bash
pytest tests/ -x --tb=short
python my_script.py
```

### Running tests

```bash
# Full suite
mamba deactivate && mamba deactivate && mamba activate gensbi
pytest tests/ -x --tb=short

# Specific file
pytest tests/recipes/test_unified_conditional_pipeline.py -x --tb=short

# With parallel workers (if pytest-xdist is available)
pytest tests/ -n 2 -x --tb=short
```

---

## Coding Conventions

### Model calling convention

All loss functions and solvers must call models with **named arguments**:

```python
# ✅ Correct — named args, order doesn't matter
model(obs=x_t, t=t, **model_extras)

# ❌ Wrong — positional args, order-dependent and fragile
model(x_t, t, **model_extras)
```

Model wrappers (`ConditionalWrapper`, `JointWrapper`, `UnconditionalWrapper`) have signature `__call__(self, t, obs, ...)`, so positional `model(x_t, t)` silently swaps `t` and `obs`.

### Uniform loss interface

All loss classes (`FMLoss`, `EDMLoss`, `SMLoss`) share this signature:

```python
loss_obj(model, batch, condition_mask=None, model_extras=None) → scalar
```

- `batch` is always `(x_0, x_1, t_or_sigma)`
- `condition_mask` is used for x_t masking AND passed to the model (for joint models)
- `model_extras` is a dict of additional kwargs forwarded to `model(**model_extras)`

### `condition_mask` handling for joint models

`condition_mask` serves **two purposes** in joint models:

1. **x_t masking:** conditioned variables are set to clean data `x_1` instead of noisy `x_t`
2. **Model input:** the model needs to know which variables are conditioned

Therefore, `condition_mask` must be included in **both** the `condition_mask` argument (for x_t masking) **and** in `model_extras` (so the model receives it):

```python
model_extras = {
    "node_ids": self.node_ids,
    "condition_mask": condition_mask,  # model needs this!
}
loss_obj(model, prepared, condition_mask=condition_mask, model_extras=model_extras)
```

---

## Debugging Tips

### Shape mismatch errors in losses

If you see errors like `sub got incompatible shapes for broadcasting: (1, 32, 1), (32, 2, 2)`, check:

1. Is the model being called with positional instead of named args?
2. Is `t` and `obs` swapped in the call?
3. Does the model expect `condition_mask` but it's not in `model_extras`?

### Test failures after renaming/moving classes

After renaming or relocating a class, always search the full codebase:

```bash
grep -r "OldClassName" src/ tests/
```

Check both source AND test files — isinstance checks and inline imports are easy to miss.
