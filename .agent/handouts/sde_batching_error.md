# Handout: SDE Solver Batch Semantics

## Intended Semantics

| Method | `x_o` shape | Output shape | Use case |
|--------|------------|--------------|----------|
| `sample(key, x_o, nsamples)` | `(1, dim_cond, ch)` | `(N, dim_obs, ch)` | N draws from p(θ\|x_o) |
| `sample_batched(key, x_o, nsamples)` | `(B, dim_cond, ch)` | `(N, B, dim_obs, ch)` | N draws for each of B conditions |

Both methods exist — `sample_batched` is inherited from `AbstractPipeline`.

## Why SDE Sampling Cannot Be Natively Batched (Like ODE)

The **ODE solver** passes the full batch directly to `diffeqsolve`:
```python
# x_init: (N, dim, ch) — diffrax integrates all N trajectories in one call
diffeqsolve(ODETerm(vector_field), ..., y0=x_init)
```
This works because the ODE is deterministic: one vector field evaluation for the whole batch.

The **SDE solver** must use `vmap(sample_one)` — each sample needs its **own independent Brownian path** with its own random key:
```python
# Per sample: independent VirtualBrownianTree with unique key_i
brownian_motion = VirtualBrownianTree(shape=(flat_dim,), key=key_i, ...)
```
`VirtualBrownianTree` is a *virtual* (compressed) representation — it cannot produce N independent paths from one key. Creating `VirtualBrownianTree(shape=(N, flat_dim))` would give N correlated (or identical) paths. Hence `vmap(sample_one)` is the correct and necessary pattern.

**Consequence**: inside `sample_one`, `obs` always has batch size 1. The model must receive `cond` with batch size 1 too.

## The Bug

`sample()` (and `sample_batched()`) do not enforce `x_o.shape[0] == 1`. When `x_o` has batch size B > 1, the condition is baked into the sampler as `(B, dim_cond, ch)`, but each `sample_one` call integrates a single sample (`obs` batch=1):

- **`JointWrapper` (line 73)**: `jnp.broadcast_to(cond, (obs.shape[0], ...))` **crashes** — cannot broadcast B → 1.
- **Real `ConditionalWrapper`**: would fail or produce wrong results silently.
- **`MockConditionalModel`**: returns `zeros_like(obs)`, never touching `cond` → mismatch invisible.

The visible diffrax error (`MultiTerm ... but expected AbstractTerm`) was a tracing failure caused by the shape mismatch, not a solver incompatibility.

`sample_batched` has the same bug: it calls `get_sampler(x_o)` with the full B-batch baked in, then calls the sampler N times — each time the SDE loop sees obs=1 vs cond=B.

## Fix Plan

### 1. Assert `x_o.shape[0] == 1` in `get_sampler`

In `ConditionalPipeline.get_sampler` and `JointPipeline.get_sampler`:
```python
cond = _expand_dims(x_o)
if cond.shape[0] != 1:
    raise ValueError(
        f"x_o must have batch size 1 for sample(). "
        f"Got {cond.shape[0]}. Use sample_batched() for multiple conditions."
    )
```

### 2. Fix `sample_batched` to loop per condition

```python
def sample_batched(self, key, x_o, nsamples, ...):
    keys = jax.random.split(key, x_o.shape[0])
    results = [
        self.get_sampler(x_o[i:i+1], ...)(k, nsamples)
        for i, k in enumerate(keys)
    ]
    return jnp.stack(results, axis=1)  # (nsamples, B, dim_obs, ch)
```

### 3. Update tests

All `x_o` in `sample()` calls → shape `(1, dim_cond, ch)` (already fixed). Remove `xfail` markers from joint tests.

## Key Files

| File | Relevance |
|------|-----------| 
| [conditional_pipeline.py:L234](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/conditional_pipeline.py#L234) | `get_sampler` — add assertion |
| [joint_pipeline.py:L360](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/joint_pipeline.py#L360) | `get_sampler` — add assertion |
| [pipeline.py:L809](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/recipes/pipeline.py#L809) | `sample_batched` — fix loop |
| [joint.py:L73](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/models/wrappers/joint.py#L73) | `broadcast_to` — fine once B=1 is enforced |
| [sde_solver_fm.py](file:///data/users/Aurelio/Github/GenSBI/src/gensbi/flow_matching/solver/sde_solver_fm.py) | `vmap(sample_one)` — no change needed |
