# Uniform `(B, dim, C)` channel convention + conditioner rename/redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the discrete-flow track (`MAFlow`/`TarFlow` + `ConditionalFlowPipeline` + `NLEPosterior`) carry a mandatory channel axis `(B, dim, C)` end-to-end for every `C ≥ 1` (no `C=1`-vs-`C>1` branching), and bring TarFlow's conditioners onto one uniform channel fold with clearer names.

**Architecture:** A normalizing flow is a bijection on ℝ^d; a size-1 channel axis is pure bookkeeping (numerically free). We (1) rename conditioner classes + de-conflate the `channels` config word **first**, (2) land the backward-compatible `AdditiveBiasConditioner` channel fold, (3) flip the whole tabular-vector stack to channel-carrying **in one coordinated task** (tokenizer → models → pipeline → `NLEPosterior` → all their tests, including 3 integration files), (4) redesign the vector conditioner per-coordinate, (5) wire + smoke-train the examples.

**Tech Stack:** Python, JAX (0.10.2), Flax NNX, pytest (9.0.2). Two repos: `GenSBI` (library) and `GenSBI-examples` (benchmark scripts).

## Global Constraints

- **Run environment:** ALL python/pytest/scripts run in the mamba env `gensbi` and on **CPU**. Prefix every command with `JAX_PLATFORMS=cpu mamba run -n gensbi …`. Never use any `.venv`.
- **Suite runs use `-m "not slow"`** unless a step says otherwise (`tests/normalizing_flows/test_flow_pipeline_e2e.py` trains 4000 steps under `@pytest.mark.slow`).
- **Branch:** `maf` (GenSBI). Commit after every task. Examples repo committed in its own task.
- **Spec:** `docs/superpowers/specs/2026-06-29-uniform-channel-convention-and-conditioner-redesign-design.md` — binding contract.
- **Universal shape contract:** `log_prob(x:(B,dim,C_obs), cond:(B,cond_dim,C_cond)) -> (B,)`; `sample(...) -> (B,dim,C_obs)`; `example_shape=(dim,C_obs)` with `C=1 → (dim,1)`, never collapsed.
- **`log_prob` returns `(B,)`**, never `(B,1)`.
- **Strict rejection of a bare `(B,dim)` lives at the PIPELINE only.** The model cannot detect it (a bare `(B,dim)` and an unbatched `(dim,1)` are both rank-2 and indistinguishable by rank). The model contract is `(B,dim,C)` or an unbatched `(dim,C)`; callers (`NLEPosterior`) must pass channel-carrying shapes.
- **Conditioner channel rule:** `cond` carries `(B,cond_dim,C_cond)` at the pipeline boundary. The `AdditiveBiasConditioner` and MAF **flatten** it (so they also tolerate a bare `(B,cond_dim)`); the per-coordinate `VectorConditioner` (Task 5) requires the channel axis. In model **unit** tests the modeled-variable (obs) input must be `(B,dim,C)`, but a `cond` fed to a bias/MAF model may stay flat `(B,cond_dim)` (it is flattened internally).
- **Dev branch:** breaking changes acceptable when mathematically correct; no checkpoint/numerical-identity gate. `C=1` data bookkeeping is byte-identical internally; the per-coordinate `VectorConditioner` is a genuine architecture change (zero blast radius — no example/test trains a prefix-conditioned model).
- **Repo paths:** GenSBI = `/lhome/ific/a/aamerio/data/github/GenSBI`; GenSBI-examples = `/lhome/ific/a/aamerio/data/github/GenSBI-examples`.

---

## File-structure map

| File | Responsibility | Tasks |
|---|---|---|
| `src/gensbi/models/tarflow/conditioners.py` | the 3 conditioner classes | 1, 3, 5 |
| `src/gensbi/models/tarflow/model.py` | TarFlowParams, make_cond, example_shape/mean/std, robust set_standardization, cond_channels wiring | 1, 3, 4, 5 |
| `src/gensbi/models/tarflow/blocks.py` | MetaBlock docstring | 1 |
| `src/gensbi/models/core/tokenizers.py` | VectorTokenizer example_shape/detokenize | 4 |
| `src/gensbi/models/maf/model.py` | MAFlow.sample / robust set_standardization | 4 |
| `src/gensbi/recipes/flow_pipeline.py` | strict channel, drop squeeze/expand, unify cond-prep, fit_standardization | 4 |
| `src/gensbi/inference/posterior.py` | channel-carrying x_o/theta into the flow | 4 |
| tests across `tests/models/{core,tarflow,maf}`, `tests/normalizing_flows` | channel-carrying shapes + renames | 1,3,4,5 |
| GenSBI-examples 3 tarflow + 3 MAF benchmarks | configs + scripts | 2, 6 |

---

## Task 1: Rename conditioners + `cond` strings (library, behaviour-preserving)

**Files:**
- Modify: `src/gensbi/models/tarflow/conditioners.py`, `model.py`, `blocks.py` (docstring)
- Test: `tests/models/tarflow/test_conditioners.py`, `test_blocks_meta.py`, `test_model.py`, `test_tarflow.py`, `test_structured_integration.py`, `test_structured_boundary.py`

**Interfaces — Produces (signatures UNCHANGED this task):**
- `AdditiveBiasConditioner(cond_dim, channels, rngs)` (was `VectorConditioner`)
- `VectorConditioner(cond_dim, channels, num_tokens, rngs)` (was `VectorPrefixConditioner`; still dense prefix here)
- `ImageConditioner(cond_channels, patch_size, channels, num_tokens, rngs)` (was `ImagePrefixConditioner`)
- `cond=` strings: `"bias"` (default), `"vector"`, `"image"`

- [ ] **Step 1: Rename the classes (SWAP — order matters)**

In `conditioners.py`, word-boundary edits IN THIS ORDER:
1. `VectorConditioner` → `AdditiveBiasConditioner`
2. `VectorPrefixConditioner` → `VectorConditioner`
3. `ImagePrefixConditioner` → `ImageConditioner`

> ⚠️ Do #1 before #2 — reversing creates two `VectorConditioner`s. `VectorPrefixConditioner` does not contain the token `VectorConditioner`, so #1 is safe.

Update each class's opening docstring sentence to its new role name.

- [ ] **Step 2: Update `model.py`**

- Imports: `from gensbi.models.tarflow.conditioners import (AdditiveBiasConditioner, VectorConditioner, ImageConditioner,)`.
- `make_cond()`: `cond=="bias"` → `AdditiveBiasConditioner(...)`; `cond=="vector"` → `VectorConditioner(...)`; `cond=="image"` → `ImageConditioner(...)` (same constructor args as today).
- `__post_init__`: validation set `("bias", "vector", "image")`; guard `if self.cond == "image" and (...)`.
- `TarFlowParams.cond` default → `"bias"`.
- Update the `cond` docstring block to the new strings/class names.

- [ ] **Step 3: Update the tests (renames only)**

- `test_conditioners.py`: import `AdditiveBiasConditioner` for the bias tests; import `VectorConditioner, ImageConditioner`; `VectorConditioner(cond_dim, channels, num_tokens=…)`, `ImageConditioner(...)`. **Rename function `test_image_prefix_shapes` → `test_image_conditioner_shapes`** (leave `test_vector_prefix_shapes` as-is; it is replaced in Task 5).
- `test_blocks_meta.py`: line 5 `import AdditiveBiasConditioner`; line 6 `import VectorConditioner`; line 14 `AdditiveBiasConditioner(cond_dim, channels, rngs=rngs)`; line 84 `VectorConditioner(cond_dim, channels, num_tokens, rngs=rngs)`.
- `test_model.py:158`, `test_tarflow.py:15`, `test_structured_integration.py:55`, `test_structured_boundary.py` (any `cond="image_prefix"`): → `cond="image"`.

- [ ] **Step 4: Run the tarflow suite green**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q -m "not slow"
```
Expected: PASS (same count as before; pure refactor).

- [ ] **Step 5: Confirm no stale references (precise grep — quoted strings only, to avoid test-function-name false positives)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && \
  grep -rn "VectorPrefixConditioner\|ImagePrefixConditioner" src/ tests/ ; \
  grep -rnE "\"add\"|'add'|\"vector_prefix\"|'vector_prefix'|\"image_prefix\"|'image_prefix'" src/ tests/
```
Expected: no output from either grep. (The bare word `VectorConditioner` now refers to the new class — intended. The function name `test_vector_prefix_shapes` is not matched by these greps.)

- [ ] **Step 6: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "refactor(tarflow): rename conditioners (AdditiveBias/Vector/Image) + cond strings

Behaviour-preserving swap rename + cond strings add/vector_prefix/image_prefix ->
bias/vector/image (default bias). Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Examples — drop the `channels:` width key (3 tarflow benchmarks)

**Files (GenSBI-examples):** `examples/sbi-benchmarks/{two_moons/tarflow, slcp/tarflow_NLE, slcp/tarflow_NPE}` — every `config/*.yaml` + the `build_flow` in each `train_tarflow_*.py`.

**Interfaces:** Produces configs specifying `num_heads:`; `build_flow` reads it. Width `head_dim·num_heads` unchanged.

- [ ] **Step 1: Edit each config — `channels:` → `num_heads:` (= old_channels // head_dim)**

Per file, delete `channels:` and add `num_heads:` with `old_channels // head_dim`. Example (`two_moons/tarflow/config/config_tarflow_npe.yaml`, `channels: 80`, `head_dim: 20`):
```yaml
model:
  num_blocks: 8
  num_heads: 4               # attention heads; total width = head_dim * num_heads
  layers_per_block: 2
  head_dim: 20
  block_size: 1
  permutation: flip
  standardize: true
  zero_init: true
```
Apply per-file (each variant has its own `channels`/`head_dim`).

- [ ] **Step 2: Edit each `build_flow` to read `num_heads` directly**

Replace the `channels`/`num_heads` block (e.g. `two_moons/tarflow/train_tarflow_npe.py:42-54`):
```python
    head_dim = int(model_cfg.get("head_dim", 16))
    num_heads = int(model_cfg.get("num_heads", 4))
    return TarFlow(TarFlowParams(
        rngs=rngs, dim=dim_obs, cond_dim=dim_cond,
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        head_dim=head_dim, num_heads=num_heads,
        block_size=int(model_cfg.get("block_size", 1)),
        permutation=str(model_cfg.get("permutation", "flip")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    ))
```
Drop the "config keeps specifying channels" sentence from the docstring.

- [ ] **Step 3: Verify each benchmark builds a model (CPU, gensbi env)**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -c "
import yaml, sys; sys.path.insert(0,'/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow')
from flax import nnx
import train_tarflow_npe as m
cfg = yaml.safe_load(open('/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow/config/config_tarflow_npe.yaml'))
print('built', type(m.build_flow(nnx.Rngs(0), 2, 3, cfg['model'])).__name__)
"
```
Expected: `built TarFlow`. Repeat for the slcp NLE/NPE scripts (adjust path/signature).

- [ ] **Step 4: Confirm no `channels:` width key remains**

```bash
grep -rn "channels" /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/{two_moons/tarflow,slcp/tarflow_NLE,slcp/tarflow_NPE}/config/
```
Expected: no `channels:` lines.

- [ ] **Step 5: Commit (examples repo)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && git add -A && \
git commit -m "refactor(tarflow examples): replace channels: width key with num_heads

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `AdditiveBiasConditioner` channel fold (backward-compatible)

Land this BEFORE the pipeline flip (Task 4): it accepts both flat `(B,cond_dim)` and `(B,cond_dim,C_cond)`, so it keeps the still-squeezing pipeline green now and the channel-carrying pipeline green later.

**Files:** `src/gensbi/models/tarflow/conditioners.py` (`AdditiveBiasConditioner`), `model.py` (`make_cond`), `tests/models/tarflow/test_conditioners.py`.

**Interfaces — Produces:** `AdditiveBiasConditioner(cond_dim, channels, rngs, cond_channels=1)`; `embed(cond) -> ((B,channels), None)`, accepting `(B,cond_dim)` or `(B,cond_dim,C_cond)`.

- [ ] **Step 1: Write the failing test**

`test_conditioners.py`:
```python
def test_additive_bias_channel_carrying_cond():
    cond_dim, cond_channels, channels, B = 3, 2, 8, 4
    c = AdditiveBiasConditioner(cond_dim, channels, rngs=nnx.Rngs(0),
                                cond_channels=cond_channels)
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim, cond_channels))
    bias, prefix = c.embed(cond)
    assert bias.shape == (B, channels) and prefix is None
```

- [ ] **Step 2: Run to verify failure**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  "/lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow/test_conditioners.py::test_additive_bias_channel_carrying_cond" -q
```
Expected: FAIL (`cond_channels` kwarg unknown / Linear shape mismatch).

- [ ] **Step 3: Implement**

In `conditioners.py` add `import jax.numpy as jnp` (currently only `import jax`), and rewrite `AdditiveBiasConditioner`:
```python
    def __init__(self, cond_dim, channels, rngs, cond_channels: int = 1):
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim * cond_channels, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond):
        if self.cond_dim == 0:
            return (None, None)
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        cond = jnp.asarray(cond).reshape(cond.shape[0], -1)    # (B, cond_dim*C_cond)
        bias = self.l2(jax.nn.silu(self.l1(cond)))
        return (bias, None)
```
In `model.py` `make_cond`, the `"bias"` branch:
```python
            if params.cond == "bias":
                return AdditiveBiasConditioner(params.cond_dim, channels, rngs=rngs,
                                               cond_channels=params.cond_channels)
```

- [ ] **Step 4: Run the tarflow suite green**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q -m "not slow"
```
Expected: PASS (existing bias tests pass flat `(B,cond_dim)`; `reshape(B,-1)` is a no-op there).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(tarflow): AdditiveBiasConditioner folds cond channel (cond_channels), backward-compatible

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Channel-carry the tabular-vector stack (coordinated)

> The tokenizer/TarFlow `example_shape`, MAF `sample`, the pipeline squeeze/expand, and `NLEPosterior` are a **matched set**: the integration tests (`test_pipeline_integration.py`, `test_structured_integration.py`, `test_structured_boundary.py`) assert end-to-end shapes through TarFlow→pipeline→posterior, so these change together. Implement production code first (Steps 1–6), then migrate tests file-by-file (Steps 7–13), then full-suite green (Step 14). There is no green sub-state between Steps 1 and 13 — that is inherent to the coupling.

**Files — Modify:** `models/core/tokenizers.py`, `models/tarflow/model.py`, `models/maf/model.py`, `recipes/flow_pipeline.py`, `inference/posterior.py`.
**Files — Test (migrate):** `tests/models/core/test_tokenizers.py`, `tests/models/tarflow/{test_model.py,test_tarflow.py,test_blocks_meta.py,test_pipeline_integration.py,test_structured_integration.py,test_structured_boundary.py}`, `tests/models/maf/{test_maflow.py,test_maflow_density.py}`, `tests/normalizing_flows/test_flow_pipeline.py`.

**Interfaces — Produces:**
- `VectorTokenizer.example_shape == (dim, channels)`; `detokenize -> (B,dim,channels)`.
- `TarFlow.example_shape == (dim, C_obs)`; robust `set_standardization` (accepts `(dim,)`/`(dim,1)`/`(C,)`/`(dim,C)`/scalar broadcastable to `example_shape`).
- `MAFlow.sample -> (B,dim,channels)`; robust `set_standardization` likewise.
- Pipeline: `_require_channel(x,name)` (rejects rank<3 tabular); `_single_obs(x_o) -> per-obs shape with channel kept`; `get_sampler`/`sample -> (nsamples,dim,C)`; `log_prob_fn((B,dim,C)) -> (B,)`; `fit_standardization` on native shape; **no** `_obs_passthrough`/`_cond_passthrough`/`_squeeze_ch`/`_expand_dims`/`_single_cond`/`_structured_cond`.
- `NLEPosterior` feeds the flow channel-carrying `x_o`/`theta`.

### Production code

- [ ] **Step 1: VectorTokenizer always `(dim, channels)`**

`tokenizers.py` `VectorTokenizer`:
```python
        self.example_shape = (dim, channels)

    def detokenize(self, tokens):
        B = tokens.shape[0]
        return tokens.reshape(B, self.dim, self.channels)
```
Update its docstrings (input `(B,dim,C)`, `C=1 → (dim,1)`; `detokenize -> (B,dim,channels)`).

- [ ] **Step 2: TarFlow robust `set_standardization`**

`tarflow/model.py` — add a helper and use it (replaces the two `broadcast_to(..., self.example_shape)` lines):
```python
    def _fit_stat(self, s, dtype):
        s = jnp.asarray(s, dtype=dtype)
        es = self.example_shape
        if s.ndim == 1 and s.shape[0] == es[0]:
            s = s.reshape((es[0],) + (1,) * (len(es) - 1))   # (dim,) -> (dim,1,...)
        return jnp.broadcast_to(s, es)

    def set_standardization(self, mean, std) -> None:
        if not self._standardize:
            raise ValueError("TarFlow built with standardize=False")
        self.mean[...] = self._fit_stat(mean, self.mean[...].dtype)
        self.std[...] = self._fit_stat(std, self.std[...].dtype)
```
(`example_shape` becomes `(dim, C_obs)` automatically from the Task-1-unchanged tokenizer wiring once Step 1 lands; no other TarFlow change is needed — `_ensure_batched`, `_base_log_prob`, `log_prob`, `sample` already work on `(B,dim,C)`.)

- [ ] **Step 3: MAF `sample` + robust `set_standardization`**

`maf/model.py`:
```python
        x = x.reshape(x.shape[0], self.dim, self.channels)   # always carry the channel
        return x
```
and
```python
    @staticmethod
    def _fit_stat(s, es):
        s = jnp.asarray(s)
        if s.ndim == 1 and s.shape[0] == es[0]:
            s = s.reshape((es[0],) + (1,) * (len(es) - 1))
        return jnp.broadcast_to(s, es)

    def set_standardization(self, mean, std) -> None:
        es = (self.dim, self.channels)
        mean = self._fit_stat(mean, es).reshape(-1)
        std = self._fit_stat(std, es).reshape(-1)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
```
Update `sample` docstring "Returns" → `(nsamples, dim, channels)` for all `C`.

- [ ] **Step 4: `NLEPosterior` feeds channel-carrying shapes**

`inference/posterior.py` `build_target`:
```python
        if self.structured_obs:
            x_o = jnp.asarray(x_o)                                   # (H, W, C)
        else:
            x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))[..., None]   # (dim_x, 1)
        ...
        def log_likelihood(theta):
            theta = jnp.asarray(theta)
            return flow.log_prob(x_o[None], theta[None, :, None])[0]   # cond (1,dim,1)
```
(For non-structured: obs `(1,dim_x,1)`; structured: obs `(1,H,W,C)`. theta/cond always `(1,dim,1)`. MAF flattens; TarFlow with `example_shape=(dim,1)` accepts it.) `sample`'s `_expand_dims(samples)` (line 137) is unchanged.

- [ ] **Step 5: Pipeline — strict channel, drop squeeze/expand, unify cond-prep, native standardization**

`recipes/flow_pipeline.py`:
- Remove `from gensbi.utils.math import _expand_dims` and the helpers `_squeeze_ch`, `_single_cond`, `_structured_cond`.
- Add:
```python
def _require_channel(x, name="input"):
    """Enforce a tabular channel axis (B, dim, C); reject a bare (B, dim)."""
    x = jnp.asarray(x)
    if x.ndim < 3:
        raise ValueError(
            f"{name} must carry a channel axis (B, dim, C); got shape "
            f"{tuple(x.shape)}. A bare (B, dim) is not accepted — add a trailing "
            f"channel axis (e.g. x[..., None] for C=1).")
    return x


def _single_obs(x_o):
    """Strip the leading batch axis from ONE observation, keeping the channel
    (and any structured) axes. Warn + take-first on a batch axis > 1."""
    x_o = jnp.asarray(x_o)
    if x_o.ndim < 2:
        raise ValueError(
            "x_o must carry a leading batch axis (e.g. (1, dim_cond, C)); got "
            f"shape {tuple(x_o.shape)}.")
    _warn_if_batched(x_o.shape[0])
    return x_o[0]
```
- In `__init__`, delete the `_obs_passthrough`/`_cond_passthrough` assignments; keep `structured_obs`/`structured_cond`, `ch_obs`/`ch_cond`.
- `_prep_obs`/`_prep_cond`:
```python
    def _prep_obs(self, x):
        x = jnp.asarray(x)
        return x if self.structured_obs else _require_channel(x, "obs")

    def _prep_cond(self, x):
        x = jnp.asarray(x)
        return x if self.structured_cond else _require_channel(x, "cond")
```
- `fit_standardization` — drop the squeeze branch:
```python
        obs = jnp.asarray(obs_data)
        mean = jnp.mean(obs, axis=axis)
        std = jnp.std(obs, axis=axis)
        std = jnp.where(std < 1e-6, 1.0, std)
        self.model.set_standardization(mean, std)
        self.ema_model.set_standardization(mean, std)
        self._standardized = True
```
- `get_sampler`:
```python
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)                          # (cond_dim, C_cond) or (H,W,C)

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
            return flow.sample(key, cond=cond_b)         # model owns (nsamples, dim, C)
        return sampler
```
- `get_log_prob_fn`:
```python
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
            return flow.log_prob(obs, cond_b)            # (B,)
        return log_prob_fn
```
- `sample_batched` is unchanged in logic (loops `get_sampler(x_o[i:i+1])`); update its docstring shape to `(nsamples, B, dim, C)`.
- Update the class + method docstrings: drop the "bare `(dim_cond,)`" promise and the `_expand_dims`/`_squeeze_ch`/`ch>1 passthrough` language; state the channel is always carried and a bare `(B,dim)` is rejected.

- [ ] **Step 6: Smoke-check the production wiring before migrating tests**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -c "
import jax, jax.numpy as jnp
from flax import nnx
from gensbi.models import TarFlow, TarFlowParams, MAFlow, MAFlowParams
tf = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, head_dim=8, num_heads=2, num_blocks=2))
print('tf example_shape', tf.example_shape)
print('tf lp', tf.log_prob(jnp.zeros((3,4,1)), jnp.zeros((3,2,1))).shape)
print('tf sample', tf.sample(jax.random.PRNGKey(0), cond=jnp.zeros((3,2,1))).shape)
tf.set_standardization(jnp.zeros(4), jnp.ones(4))   # (dim,) still accepted
mf = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=2))
print('mf sample', mf.sample(jax.random.PRNGKey(0), cond=jnp.zeros((3,2,1))).shape)
mf.set_standardization(jnp.zeros((4,1)), jnp.ones((4,1)))
print('OK')
"
```
Expected: `tf example_shape (4, 1)`, `tf lp (3,)`, `tf sample (3, 4, 1)`, `mf sample (3, 4, 1)`, `OK`.

### Test migration

**Migration rule (apply mechanically):** a modeled-variable (obs) input fed to a model/pipeline must be `(B,dim,C)` — add a trailing `1` to every bare `(B,dim)` vector obs and every `(B,dim)` sample-shape assertion. A `cond` fed directly to a TarFlow built with `cond="bias"` (or to MAF) may stay flat `(B,cond_dim)` (it is flattened). Image obs/cond already carry channels. `set_standardization` still accepts a bare `(dim,)` stat (robust setter), but assertions comparing `flow.mean[...]` (now `(dim,1)`) must compare `.ravel()` or the `(dim,1)` shape.

- [ ] **Step 7: `tests/models/core/test_tokenizers.py`**

```python
def test_shapes_scalar_per_token():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert (tok.T, tok.F) == (6, 1)
    assert tok.tokenize(jnp.arange(12.0).reshape(2, 6, 1)).shape == (2, 6, 1)

def test_vector_tokenizer_example_shape():
    assert VectorTokenizer(dim=6, block_size=1).example_shape == (6, 1)

def test_vector_tokenizer_channels_one_unchanged():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert tok.F == 2 and tok.T == 3 and tok.example_shape == (6, 1)
    x = jnp.arange(2 * 6).reshape(2, 6, 1).astype(jnp.float32)
    assert jnp.array_equal(tok.detokenize(tok.tokenize(x)), x)
```
Gate: `pytest tests/models/core/test_tokenizers.py -q` → PASS.

- [ ] **Step 8: `tests/models/tarflow/test_model.py`** (add trailing `1` to every vector obs; keep `cond` flat for the bias path; fix the two base-comparison + set_standardization assertions)

Concrete edits:
- `test_log_prob_shape_and_finite`: `x = …(8, 4, 1)`.
- `test_zero_init_flow_is_standard_normal`: keep `x = …(8, dim)` for the base; feed the flow `x[..., None]`: `lp = flow.log_prob(x[..., None], cond)`; `lp_base = jax.vmap(base.log_prob)(x)`.
- `test_full_flow_logdet_matches_autodiff`: make `x` channel-carrying — `x = jnp.array([0.5,-1.0,0.3,0.8])[:, None]` and in `to_noise` use `flow.tokenizer.tokenize(x[None])` (now `(1,4,1)`); `flow.log_prob(x[None], cond[None])` with `x[None]=(1,4,1)`.
- `test_sample_shape_and_roundtrip_finite`: `assert s.shape == (5, 4, 1)`; `flow.log_prob(s, cond)` (s now `(5,4,1)`).
- `test_density_integrates_to_one_2d`: feed `flow.log_prob(grid[..., None], cond)` (grid stays `(N,2)` for the integral math).
- `test_log_prob_depends_on_condition`: `x = …(5, 4, 1)`.
- `test_unconditional_flow`: `x = …(6, 3, 1)`.
- `test_set_standardization`: keep `mean`/`std` as `(4,)`; change asserts to `assert flow.mean[...].shape == (4, 1)` and `assert jnp.allclose(flow.mean[...].ravel(), mean)` (same for std).
- `test_set_standardization_raises_when_disabled`: unchanged (raises before broadcast).
- `test_image_modeled_*`, `test_image_set_standardization_shape`: unchanged (image already `(B,H,W,C)`).
- `test_image_condition_npe_depends_on_condition`: `theta = …(5, 2, 1)`; `assert s.shape == (5, 2, 1)`.
- `test_vector_path_unchanged`: `x = …(8, 4, 1)`.

Gate: `pytest tests/models/tarflow/test_model.py -q` → PASS.

- [ ] **Step 9: `tests/models/tarflow/test_tarflow.py`**

`test_tarflow_log_prob_and_sample_shapes`: `x = …(3,4,1)`; `assert s.shape == (3, 4, 1)`. (cond stays `(3,2)`.)
Gate: `pytest tests/models/tarflow/test_tarflow.py -q` → PASS.

- [ ] **Step 10: `tests/models/tarflow/test_blocks_meta.py`**

Only `test_tarflow_vector_channels_one_unchanged` needs the obs channel:
```python
    x = jnp.zeros((3, 4, 1))
    assert m.log_prob(x).shape == (3,)
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4, 1)
```
The `MetaBlock`-level tests pass `(B,T,F)` tokens directly and are unaffected. (The `_make_prefix` block tests still use the dense `VectorConditioner(cond_dim, channels, num_tokens, rngs)` and flat cond — they are migrated in Task 5.)
Gate: `pytest tests/models/tarflow/test_blocks_meta.py -q` → PASS.

- [ ] **Step 11: `tests/models/maf/test_maflow.py` + `test_maflow_density.py`**

- `test_maflow.py::test_maflow_sample_shape_and_standardize`:
```python
    c = jax.random.normal(jax.random.key(0), (7, 2, 1))
    assert flow.sample(jax.random.key(1), cond=c).shape == (7, 4, 1)
    assert flow.log_prob(jnp.zeros((5, 4, 1)), jnp.zeros((5, 2, 1))).shape == (5,)
    flow.set_standardization(jnp.ones((4, 1)), 2.0 * jnp.ones((4, 1)))
```
- `test_maflow_density.py`: add a trailing `1` to bare `(B,dim)` sample-shape assertions (e.g. `test_sample_shape_and_roundtrip_consistency` → `(5,3,1)`). `test_set_standardization_sets_buffers` (passes a `(3,)` stat) is unchanged — the robust setter accepts `(dim,)`; if it asserts buffer shape, compare `.ravel()` (MAF stores the flat `(dim*channels,)` buffer, so a `(3,)` comparison still holds for C=1).

Gate: `pytest tests/models/maf -q` → PASS.

- [ ] **Step 12: `tests/normalizing_flows/test_flow_pipeline.py`**

- Imports (lines 12-14): `from gensbi.recipes.flow_pipeline import (ConditionalFlowPipeline, _require_channel, _single_obs,)`.
- Replace `test_squeeze_ch`:
```python
def test_require_channel_rejects_bare_2d():
    assert _require_channel(jnp.zeros((4, DIM_OBS, 1)), "obs").shape == (4, DIM_OBS, 1)
    with pytest.raises(ValueError):
        _require_channel(jnp.zeros((4, DIM_OBS)), "obs")
```
- Replace `test_single_cond`, `test_single_cond_batched_warns_and_takes_first`, `test_structured_cond_strips_only_batch_axis`:
```python
def test_single_obs_keeps_channel_strips_batch():
    assert _single_obs(jnp.zeros((1, DIM_COND, 1))).shape == (DIM_COND, 1)
    img = jnp.arange(1*1*4*2).reshape(1, 1, 4, 2)
    assert _single_obs(img).shape == (1, 4, 2)               # H==1 preserved

def test_single_obs_batched_warns_and_takes_first():
    x_o = jnp.arange(3 * DIM_COND).reshape(3, DIM_COND, 1)
    with pytest.warns(UserWarning, match="batch dimension"):
        out = _single_obs(x_o)
    assert out.shape == (DIM_COND, 1) and jnp.array_equal(out, x_o[0])
```
- Kwarg tests (~227, 233, 239, 245): change `jnp.zeros((1, DIM_COND))` → `jnp.zeros((1, DIM_COND, 1))` (and the `sample_batched` one to `(2, DIM_COND, 1)`).
- `test_multichannel_both_sample_and_logprob_shapes`: DELETE the two asserts `assert pipe._obs_passthrough` / `assert pipe._cond_passthrough` (the attributes are removed); keep the shape asserts.
- `test_fit_standardization_sets_both_models` (MAF, ~line 119): unchanged — MAF stores the flat `(dim,)` buffer, so the `(DIM,)` comparison still holds.

Gate: `pytest tests/normalizing_flows/test_flow_pipeline.py -q` → PASS.

- [ ] **Step 13: the three TarFlow→pipeline integration files**

- `test_pipeline_integration.py`: data is already `(N,M+D,1)`; `x_1`/`x_o` already channel-carrying. Only fix `test_fit_standardization_sets_both_models`: TarFlow stores `(M,1)`, so change `exp_mean = jnp.mean(DATA[:800, :M], axis=0)` and `exp_std = jnp.std(DATA[:800, :M], axis=0)` (drop the `, 0`) so both sides are `(M,1)`. `test_nle_log_posterior_value_and_grad` passes once `posterior.py` (Step 4) lands — no test edit.
- `test_structured_integration.py`: `test_field_nle_train_smoke_and_mclmc` — cond is `_theta (N,D)` with `structured_cond=False` → feed `(N,D,1)`: change the dataset map to `(obs[i], cond[i][..., None])` **or** define a channel-carrying `_theta_c = _theta[..., None]` and use it for the cond stream. `test_image_npe_train_smoke_and_sample` — obs is `_theta (N,D)` (modeled vector) → feed `(N,D,1)` (channel-carrying obs stream); `pipe.fit_standardization(_theta[..., None])`; `pipe.log_prob(_theta[:5][..., None], _x[0:1])`; `assert s.shape == (16, D, 1)`.
- `test_structured_boundary.py`: `test_field_loss_finite_and_grads` — `cond = jnp.asarray(_theta[:32])[..., None]` (`(32,D,1)`). `test_field_fit_standardization_image_shape` and `test_field_nle_log_posterior_structured_xo` — unchanged (image obs; the posterior fix handles the `(1,D,1)` cond).

Gate: `pytest tests/models/tarflow/test_pipeline_integration.py tests/models/tarflow/test_structured_integration.py tests/models/tarflow/test_structured_boundary.py -q` → PASS.

- [ ] **Step 14: Full suite + inference green; commit**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests -q -m "not slow"
```
Expected: PASS (whole suite, including `tests/inference`).
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(flow): uniform (B,dim,C) channel end-to-end (tokenizer/MAF/TarFlow/pipeline/NLEPosterior)

Channel always carried (C=1 -> (dim,1)); pipeline rejects bare (B,dim), drops
squeeze/expand, unifies cond-prep; robust set_standardization; NLEPosterior feeds
channel-carrying x_o/theta. Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Per-coordinate `VectorConditioner` redesign (+ ImageConditioner verify)

**Files:** `conditioners.py` (`VectorConditioner`), `model.py` (`make_cond` `"vector"`, remove `prefix_tokens`), `tests/models/tarflow/{test_conditioners.py, test_blocks_meta.py}`.

**Interfaces — Produces:** `VectorConditioner(cond_dim, cond_channels, channels, rngs)`; `embed((B,cond_dim,C_cond)) -> (None, (B,cond_dim,channels))` (`M = cond_dim`, no flatten). `prefix_tokens` removed from `TarFlowParams`.

- [ ] **Step 1: Write the failing tests**

In `test_conditioners.py` replace `test_vector_prefix_shapes` and `test_prefix_depends_on_condition`:
```python
def test_vector_conditioner_per_coordinate_tokens():
    cond_dim, cond_channels, channels, B = 3, 2, 8, 4
    c = VectorConditioner(cond_dim, cond_channels, channels, rngs=nnx.Rngs(0))
    assert c.M == cond_dim
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim, cond_channels))
    bias, prefix = c.embed(cond)
    assert bias is None and prefix.shape == (B, cond_dim, channels)

def test_vector_conditioner_depends_on_condition():
    c = VectorConditioner(3, 1, 8, rngs=nnx.Rngs(0))
    _, p1 = c.embed(jnp.zeros((2, 3, 1)))
    _, p2 = c.embed(jnp.ones((2, 3, 1)))
    assert not jnp.allclose(p1, p2)
```

- [ ] **Step 2: Run to verify failure**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow/test_conditioners.py -q -k vector_conditioner
```
Expected: FAIL (old `(cond_dim, channels, num_tokens, rngs)` signature).

- [ ] **Step 3: Implement**

`conditioners.py` — rewrite `VectorConditioner`:
```python
class VectorConditioner(nnx.Module):
    """Embed a vector condition as one prefix token per coordinate.

    Each of the ``cond_dim`` coordinates is a token of ``C_cond`` channels; a
    shared ``Linear(cond_channels, channels)`` projects each to the transformer
    width, plus per-coordinate positional embeddings. Produces ``M = cond_dim``
    prefix tokens (no flatten).
    """

    def __init__(self, cond_dim, cond_channels, channels, rngs):
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels
        self.channels = channels
        self.M = cond_dim
        self.proj = nnx.Linear(cond_channels, channels, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (cond_dim, channels)) * 1e-2)

    def embed(self, cond):
        if cond is None:
            raise ValueError("cond is required for VectorConditioner")
        cond = jnp.asarray(cond)                       # (B, cond_dim, C_cond)
        return (None, self.proj(cond) + self.pos[...][None])
```
`model.py`:
- `make_cond` `"vector"`: `return VectorConditioner(params.cond_dim, params.cond_channels, channels, rngs=rngs)`.
- Remove the `prefix_tokens` field + docstring entry from `TarFlowParams`.

- [ ] **Step 4: Migrate the block-level prefix tests to channel-carrying cond**

In `test_blocks_meta.py`:
```python
def _make_prefix(T=4, F=1, channels=8, cond_dim=2, cond_channels=1, zero_init=False,
                 rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)
    cond = VectorConditioner(cond_dim, cond_channels, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     inv_perm=jnp.argsort(perm), conditioner=cond, num_layers=2,
                     num_heads=2, expansion=2, rngs=rngs, zero_init=zero_init)
```
In each `test_prefix_*`, give the cond a channel axis: unbatched jacrev cases `cond = jnp.array([0.3, -0.4])[:, None]` → `(2,1)` and pass `cond[None]` → `(1,2,1)`; batched cases `jax.random.normal(key, (3, 2, 1))`; the `test_prefix_conditions_output` two-condition case `jnp.broadcast_to(jnp.array([0.3,-0.4])[:, None], (2, 2, 1))` etc. (The `MetaBlock`/attention code is unchanged; only the conditioner input rank grows by one.)

- [ ] **Step 5: Run the tarflow suite green; confirm ImageConditioner already folds**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q -m "not slow"
```
Expected: PASS. `ImageConditioner` already folds the channel via `patchify_2d` (`in_f = cond_channels·patch²`) — `test_image_conditioner_shapes` confirms it; no change beyond the Task-1 rename.

- [ ] **Step 6: Confirm `prefix_tokens` fully removed**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && grep -rn "prefix_tokens" src/ tests/
```
Expected: no output.

- [ ] **Step 7: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(tarflow): per-coordinate VectorConditioner (no flatten, M=cond_dim); drop prefix_tokens

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Examples — channel-carrying inference + smoke-train all 6

**Files (GenSBI-examples):** the 3 tarflow benchmark scripts/configs + the 3 MAF benchmark scripts (`two_moons/maf_NPE`, `two_moons/maf_NLE`, `slcp/maf_NLE`).

- [ ] **Step 1: Audit each script's inference `x_o`/`x_1` shapes**

```bash
grep -rn "\.sample(\|\.log_prob(\|x_o\|build_target\|\[None\]\|\[\.\.\., None\]" \
  /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/{two_moons/tarflow,slcp/tarflow_NLE,slcp/tarflow_NPE,two_moons/maf_NPE,two_moons/maf_NLE,slcp/maf_NLE}/*.py
```
The dataset already emits `(B,dim,1)` (sbibm-jax collate adds `[...,None]`), so `x_o = data_x[idx][None]` is already `(1,dim,1)`. Fix any site that squeezes `x_o` to `(1,dim)`/`(dim,)` or passes a bare `(B,dim)` `x_1` to `pipeline.log_prob`. For `NLEPosterior`, `x_o` may be passed as the raw `(dim_x,)`/`(dim_x,1)` — `build_target` now adds the channel.

- [ ] **Step 2: Set data-channel params for any `C>1` benchmark**

The 3 named SBI benchmarks are `C=1` (no change beyond Step 1). If a benchmark has `C_obs>1`/`C_cond>1`, pass `vec_channels`/`cond_channels` in `build_flow`'s `TarFlowParams(...)` (or `channels`/`cond_channels` in `MAFlowParams`) and matching `ch_obs`/`ch_cond` to the pipeline.

- [ ] **Step 3: Smoke-train each of the 6 scripts a few steps (CPU)**

Per benchmark, run its trainer with a tiny step budget (`nsteps≈3` via the script's CLI/override or a temp config). Example:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python \
  /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow/train_tarflow_npe.py \
  <args to set nsteps=3 / tiny config>
```
Expected per script: training loop runs, loss finite, sampling returns `(nsamples, dim, 1)`, no shape errors. Record the final line of each run (6 runs).

- [ ] **Step 4: Full library regression (CPU)**

```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests -q -m "not slow"
```
Expected: PASS.

- [ ] **Step 5: Commit (examples repo)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && git add -A && \
git commit -m "feat(examples): channel-carrying inference for the 6 SBI flow benchmarks; smoke-trained on CPU

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** universal contract → Tasks 4 (`log_prob->(B,)`, `sample->(B,dim,C)`, reject bare 2-D) ✓; Part 1a rename → Task 1 ✓; Part 1b examples key → Task 2 ✓; Part 2.1 VectorTokenizer → Task 4 ✓; Part 2.2 TarFlow → Task 4 ✓; Part 2.3 MAF → Task 4 ✓; Part 3 AdditiveBias fold → Task 3, per-coordinate VectorConditioner + `prefix_tokens` removal → Task 5, ImageConditioner verify → Task 5 ✓; Part 4 pipeline → Task 4 ✓; `NLEPosterior` contract fix → Task 4 ✓; Part 5 tests → folded per task ✓; Part 6 examples smoke-train → Task 6 ✓.

**Reviewer findings addressed:** B1 (false grep) → Task 1 Step 1 removed; the 3 integration files explicitly migrated in Task 4 Step 13. B2/B3 (TarFlow/structured red) → coordinated Task 4 + robust `set_standardization`. B4 (`_obs_passthrough` assert) → Task 4 Step 12 deletes lines 327-328. B5 (Task4→5 ordering) → `AdditiveBiasConditioner` fold moved to Task 3 (before the pipeline flip). B6 (`NLEPosterior` bare 2-D) → Task 4 Step 4 fixes `posterior.py`; model-level rejection rejected as infeasible (rank ambiguity), pipeline-level rejection retained. B7 (MAF `(dim,)` stat) → robust `set_standardization`. M1 (Task 1 grep false positives) → quoted-string grep + `test_image_prefix_shapes` renamed. M2/M3 (dead `ch_*`, docstring drift) → Task 4 Step 5 updates docstrings; `ch_obs/ch_cond` kept as informational.

**Placeholder scan:** production steps show full code; test-migration steps list concrete per-file edits + a per-file pytest gate. Task 6 Step 3's per-script arg adaptation is unavoidable (scripts differ) but states the exact command shape + pass criterion.

**Type consistency:** `_require_channel`/`_single_obs` defined and used consistently (Task 4); `VectorConditioner(cond_dim, cond_channels, channels, rngs)` (Task 5) matches `make_cond` + `_make_prefix`; `AdditiveBiasConditioner(cond_dim, channels, rngs, cond_channels=1)` (Task 3) matches `make_cond` + block tests; `_fit_stat` helper consistent across MAF/TarFlow; `cond` strings `bias/vector/image` consistent.

## Execution Handoff

Choose how to execute (see end of message).
