# Mixed-Precision Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** fp32 master weights + per-model bf16 compute knob across all GenSBI models, with fp32 islands (norms, softmax, time embeddings, final projection, loss), fixing the bf16 EMA-accumulation bug.

**Architecture:** Every `*Params` dataclass gains a `dtype` (compute) field and flips `param_dtype` to fp32. Both are threaded into every `nnx.Linear`/norm/conv. Norm layers and final projections are pinned to `dtype=jnp.float32` (fp32 islands); models therefore **emit fp32**. Losses defensively upcast to fp32. Gradients/AdamW/EMA become fp32 automatically because JAX gives gradients the params' dtype.

**Tech Stack:** JAX 0.10.2, flax nnx 0.12.7, optax, pytest (spec: `docs/superpowers/specs/2026-07-20-mixed-precision-design.md`)

## Global Constraints

- **Test command prefix (GPU-less nodes):** `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest` — use the mamba `gensbi` env, NOT `.venv`. pyproject forces xdist `-n 2`; add `-n 0` if you need `-s` prints.
- **Contract: models emit fp32** — implemented as the final projection layer constructed with `dtype=jnp.float32`, never as a post-hoc `.astype`.
- **fp32 islands (always, regardless of the knob):** norm layers (`dtype=jnp.float32`), attention softmax (`jax.nn.dot_product_attention` already does this internally — verified in jax 0.10.2: logits einsum uses `preferred_element_type=float32` and softmax runs on fp32; `flax nnx.MultiHeadAttention` does NOT, see Task 6), timestep/sinusoidal embeddings, RoPE (already fp32), final projection, loss.
- **Defaults:** `param_dtype=jnp.float32` everywhere. `dtype=jnp.bfloat16` for Flux1, Flux1Joint, Simformer, PixelDiT, FieldDiT, autoencoders; `dtype=jnp.float32` for MAF and TarFlow.
- **MAF/TarFlow are a pure refactor:** with the knob at its fp32 default, outputs must be bit-identical to current code.
- **Inputs stay fp32:** replace every `jnp.asarray(x, dtype=self.params.param_dtype)` input cast with `jnp.asarray(x, dtype=jnp.float32)`.
- **No loss scaling** (bf16 shares fp32's exponent range).
- **Canonical threading pattern** (repeat everywhere):
  ```python
  # compute layer:
  nnx.Linear(..., dtype=dtype, param_dtype=param_dtype, rngs=rngs)
  # fp32 island (norms, final projection):
  nnx.LayerNorm(..., dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
  ```
- Commit after every task; prefixes `feat:`/`fix:`/`test:` as appropriate.
- `gensbi.models.healswin` is explicitly OUT OF SCOPE (external `heal-swin-nnx` package).

---

### Task 1: Precision test helpers

**Files:**
- Create: `tests/precision_utils.py`
- Test: `tests/test_precision_utils.py`

**Interfaces:**
- Produces: `assert_tree_dtype(tree, dtype)` — asserts every floating-point leaf of a pytree (e.g. `nnx.to_pure_dict(nnx.state(model, nnx.Param))`) has exactly `dtype`; raises `AssertionError` listing offending paths. `float_leaves(tree)` — returns `{joined_path: dtype}` for floating leaves. All later model tasks import these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precision_utils.py
import jax.numpy as jnp
import pytest
from flax import nnx

from tests.precision_utils import assert_tree_dtype, float_leaves


def test_assert_tree_dtype_passes_on_uniform_tree():
    tree = {"a": jnp.ones((2,), jnp.float32), "b": {"c": jnp.zeros((3,), jnp.float32)}}
    assert_tree_dtype(tree, jnp.float32)


def test_assert_tree_dtype_ignores_non_float_leaves():
    tree = {"ids": jnp.zeros((2,), jnp.int32), "w": jnp.ones((2,), jnp.float32)}
    assert_tree_dtype(tree, jnp.float32)


def test_assert_tree_dtype_fails_and_names_offender():
    tree = {"good": jnp.ones((2,), jnp.float32), "bad": jnp.ones((2,), jnp.bfloat16)}
    with pytest.raises(AssertionError, match="bad"):
        assert_tree_dtype(tree, jnp.float32)


def test_float_leaves_reports_dtypes():
    tree = {"w": jnp.ones((2,), jnp.bfloat16)}
    assert float_leaves(tree) == {"w": jnp.bfloat16}


def test_works_on_nnx_param_state():
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    tree = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    assert_tree_dtype(tree, jnp.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_precision_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.precision_utils'`

- [ ] **Step 3: Write the implementation**

```python
# tests/precision_utils.py
"""Shared dtype-assertion helpers for mixed-precision tests."""

import jax.numpy as jnp
import flax.traverse_util as tu


def float_leaves(tree):
    """Return {'.'-joined path: dtype} for every floating-point leaf."""
    flat = tu.flatten_dict(tree)
    return {
        ".".join(str(p) for p in path): leaf.dtype
        for path, leaf in flat.items()
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    }


def assert_tree_dtype(tree, dtype):
    """Assert every floating leaf of ``tree`` has exactly ``dtype``."""
    offenders = {k: d for k, d in float_leaves(tree).items() if d != jnp.dtype(dtype)}
    assert not offenders, (
        f"expected all floating leaves to be {jnp.dtype(dtype)}, got: {offenders}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_precision_utils.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/precision_utils.py tests/test_precision_utils.py
git commit -m "test: add shared dtype-assertion helpers for mixed-precision work"
```

---

### Task 2: Loss functions upcast to fp32

**Files:**
- Modify: `src/gensbi/flow_matching/loss/fm_loss.py` (FMLoss.__call__, line ~66)
- Modify: `src/gensbi/diffusion/loss/edm_loss.py`, `src/gensbi/diffusion/loss/sm_loss.py` (same pattern: cast model output and target to fp32 before weighting/reduction)
- Modify: `src/gensbi/recipes/flow_pipeline.py` (the NF `loss_fn` returning `-log_prob`: wrap final scalar in `jnp.asarray(..., jnp.float32)` — locate via `grep -n log_prob src/gensbi/recipes/flow_pipeline.py`)
- Test: `tests/models/losses/test_loss_fp32.py`

**Interfaces:**
- Produces: every loss callable returns an fp32 scalar even when the model emits bf16. No signature changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/losses/test_loss_fp32.py
import jax
import jax.numpy as jnp

from gensbi.flow_matching.loss import FMLoss
from gensbi.flow_matching.path import CondOTProbPath  # adjust import to the
# AffineProbPath subclass used in existing tests under tests/flow_matching/


class _Bf16Model:
    """Fake velocity model that emits bf16, violating the emit-fp32 contract."""

    def __call__(self, obs, t, **kwargs):
        return jnp.asarray(obs, jnp.bfloat16) * jnp.bfloat16(0.5)


def test_fmloss_returns_fp32_for_bf16_model():
    path = CondOTProbPath()
    loss = FMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3))
    x1 = jax.random.normal(key, (4, 3))
    t = jnp.full((4,), 0.5)
    out = loss(_Bf16Model(), (x0, x1, t))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)
```

Add one analogous test per diffusion loss (`EDMLoss`, score-matching loss), mirroring how those losses are constructed in the existing tests under `tests/diffusion/` (reuse their fixtures/constructor arguments; only the assertion `out.dtype == jnp.float32` with a bf16-emitting fake model is new).

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/losses/test_loss_fp32.py -v`
Expected: FAIL with `assert dtype(bfloat16) == float32`

- [ ] **Step 3: Implement the upcast in FMLoss**

In `fm_loss.py`, after the model call (line ~66):

```python
        model_output = model(obs=x_t, t=path_sample.t, **model_extras)

        # Loss is always computed in fp32 regardless of the model's compute
        # dtype (defense-in-depth on top of the models-emit-fp32 contract).
        model_output = jnp.asarray(model_output, jnp.float32)
        dx_t = jnp.asarray(path_sample.dx_t, jnp.float32)
```

and use `dx_t` in the squared error (`jnp.square(model_output - dx_t)`). Apply the same two-line cast (output + target) in `edm_loss.py` and `sm_loss.py` before their weighting/reduction, and cast the NF pipeline loss scalar to fp32.

- [ ] **Step 4: Run tests to verify they pass, plus existing loss tests**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/losses/ tests/flow_matching/ tests/diffusion/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/gensbi/flow_matching/loss src/gensbi/diffusion/loss src/gensbi/recipes/flow_pipeline.py tests/models/losses/
git commit -m "feat: compute all losses in fp32 regardless of model compute dtype"
```

---

### Task 3: Shared embedding modules accept a compute dtype

**Files:**
- Modify: `src/gensbi/models/embedding/embedding.py` (MLPEmbedder, GaussianFourierEmbedding, FeatureEmbedder, Embed)
- Test: `tests/models/embedding/test_embedding_dtype.py`

**Interfaces:**
- Produces: `MLPEmbedder(..., dtype=jnp.float32, param_dtype=jnp.float32)`, `FeatureEmbedder(..., dtype=jnp.float32, param_dtype=jnp.float32)` — new keyword `dtype` with fp32 default (neutral: existing callers unchanged), forwarded to internal `nnx.Linear`/`nnx.Embed`. Sinusoidal PE matrices stay fp32 (they are fp32 islands; downstream layers downcast).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/embedding/test_embedding_dtype.py
import jax.numpy as jnp
from flax import nnx

from gensbi.models.embedding import FeatureEmbedder
from gensbi.models.embedding.embedding import MLPEmbedder
from tests.precision_utils import assert_tree_dtype


def test_mlpembedder_bf16_compute_fp32_params():
    m = MLPEmbedder(4, 8, rngs=nnx.Rngs(0), dtype=jnp.bfloat16,
                    param_dtype=jnp.float32)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
    out = m(jnp.ones((2, 3, 4), jnp.float32))
    assert out.dtype == jnp.bfloat16


def test_featureembedder_absolute_bf16():
    m = FeatureEmbedder(num_embeddings=5, hidden_size=8, kind="absolute",
                        dtype=jnp.bfloat16, param_dtype=jnp.float32,
                        rngs=nnx.Rngs(0))
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
    out = m(jnp.zeros((2, 3, 1), jnp.int32))
    assert out.dtype == jnp.bfloat16


def test_default_dtype_is_neutral_fp32():
    m = MLPEmbedder(4, 8, rngs=nnx.Rngs(0))
    out = m(jnp.ones((2, 3, 4), jnp.float32))
    assert out.dtype == jnp.float32
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/embedding/test_embedding_dtype.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'dtype'`

- [ ] **Step 3: Implement**

Add `dtype: DTypeLike = jnp.float32` keyword to `MLPEmbedder.__init__`, `GaussianFourierEmbedding.__init__`, `FeatureEmbedder.__init__` (and pass through `Embed` → `nnx.Embed`), forwarding `dtype=dtype` into every `nnx.Linear`/`nnx.Embed` construction. Do NOT pass it to `SinusoidalPosEmbed1D/2D` (their precomputed `PEMatrix` tables stay at `param_dtype`, fp32 by default — sinusoidal tables are an fp32 island; `FeatureEmbedder` simply ignores `dtype` for those kinds). `MLPEmbedder.p_skip` stays at `param_dtype`.

- [ ] **Step 4: Run tests + existing embedding tests**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/embedding/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/models/embedding/embedding.py tests/models/embedding/test_embedding_dtype.py
git commit -m "feat: thread compute dtype through shared embedding modules"
```

---

### Task 4: Flux1 — params, layers, model

**Files:**
- Modify: `src/gensbi/models/flux1/model.py` (Flux1Params line ~138; Flux1.__init__ constructors; input casts lines 366-368)
- Modify: `src/gensbi/models/flux1/layers.py` (MLPEmbedder, QKNorm, SelfAttention, Modulation, DoubleStreamBlock, SingleStreamBlock, LastLayer)
- Test: `tests/models/flux1/test_flux1_precision.py`

**Interfaces:**
- Consumes: `tests.precision_utils.assert_tree_dtype` (Task 1).
- Produces: `Flux1Params(param_dtype=jnp.float32, dtype=jnp.bfloat16)` defaults; every layer class in `flux1/layers.py` accepts `dtype: DTypeLike = jnp.bfloat16, param_dtype: DTypeLike = jnp.float32` (defaults flipped from today's bf16 param_dtype); `Flux1.__call__` output dtype is fp32. Task 5 (flux1joint) reuses these layer signatures verbatim.

- [ ] **Step 1: Write the failing tests**

Reuse the smallest model construction already used in `tests/models/flux1/` (check its conftest/fixtures first and copy the smallest `Flux1Params` instantiation; the one below is a fallback):

```python
# tests/models/flux1/test_flux1_precision.py
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import Flux1, Flux1Params
from tests.precision_utils import assert_tree_dtype


def _make(dtype=jnp.bfloat16):
    params = Flux1Params(
        in_channels=1, vec_in_dim=None, context_in_dim=1, mlp_ratio=2.0,
        num_heads=2, depth=1, depth_single_blocks=1, qkv_bias=False,
        rngs=nnx.Rngs(0), dim_obs=3, dim_cond=4, axes_dim=[4],
        dtype=dtype,
    )
    return Flux1(params)


def _inputs():
    obs = jnp.ones((2, 3, 1), jnp.float32)
    cond = jnp.ones((2, 4, 1), jnp.float32)
    obs_ids = jnp.tile(jnp.arange(3)[None, :, None], (2, 1, 1))
    cond_ids = jnp.tile(jnp.arange(4)[None, :, None], (2, 1, 1))
    t = jnp.full((2,), 0.5)
    return dict(t=t, obs=obs, obs_ids=obs_ids, cond=cond, cond_ids=cond_ids)


def test_master_weights_fp32_with_bf16_compute():
    model = _make(jnp.bfloat16)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(model, nnx.Param)), jnp.float32)


def test_output_is_fp32():
    assert _make(jnp.bfloat16)(**_inputs()).dtype == jnp.float32


def test_grads_are_fp32():
    model = _make(jnp.bfloat16)
    def loss_fn(m):
        return jnp.mean(jnp.square(m(**_inputs())))
    grads = nnx.grad(loss_fn)(model)
    assert_tree_dtype(nnx.to_pure_dict(grads), jnp.float32)


def test_bf16_close_to_fp32():
    m32, m16 = _make(jnp.float32), _make(jnp.bfloat16)
    # same rngs seed -> identical fp32 master weights
    o32, o16 = m32(**_inputs()), m16(**_inputs())
    assert o16.dtype == jnp.float32
    err = jnp.max(jnp.abs(o32 - o16)) / (jnp.max(jnp.abs(o32)) + 1e-6)
    assert err < 2e-2, f"bf16 compute deviates {err} from fp32"
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/flux1/test_flux1_precision.py -v`
Expected: FAIL — `Flux1Params` has no field `dtype`.

- [ ] **Step 3: Implement**

In `Flux1Params`:
```python
    param_dtype: DTypeLike = jnp.float32
    dtype: DTypeLike = jnp.bfloat16
```
(docstring: `param_dtype` = master-weight storage, `dtype` = compute/matmul dtype).

In `flux1/layers.py`, change every class signature default from `param_dtype: DTypeLike = jnp.bfloat16` to `param_dtype: DTypeLike = jnp.float32` and add `dtype: DTypeLike = jnp.bfloat16`. Then:

- **Compute layers** (all `nnx.Linear` in MLPEmbedder, SelfAttention.qkv/proj, Modulation.lin, DoubleStreamBlock mlps, SingleStreamBlock.linear1/linear2, LastLayer.adaLN_modulation): add `dtype=dtype`.
- **fp32 islands:** every `nnx.LayerNorm` and `nnx.RMSNorm` (QKNorm, obs/cond_norm1/2, pre_norm, norm_final) gets `dtype=jnp.float32` (NOT the knob). In `QKNorm.__call__`, cast the fp32-normalized q/k back to the compute dtype so `jax.nn.dot_product_attention` sees uniform dtypes:
  ```python
  def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
      q = self.query_norm(q).astype(v.dtype)
      k = self.key_norm(k).astype(v.dtype)
      return q, k
  ```
- **LastLayer.linear** (the final projection): `dtype=jnp.float32` — this is the models-emit-fp32 contract. `norm_final` fp32 as above; `adaLN_modulation` stays at compute dtype.
- `timestep_embedding` needs no change (frequencies already fp32; with fp32 `t` the output stays fp32).

In `Flux1.__init__`: pass `dtype=params.dtype` alongside `param_dtype=params.param_dtype` into every constructed layer (obs_in, cond_in, time_in, vector_in, FeatureEmbedders, double/single blocks, final_layer). In `Flux1.__call__` replace the three input casts (lines 366-368) with:
```python
        obs = jnp.asarray(obs, dtype=jnp.float32)
        cond = jnp.asarray(cond, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
```
Attention (`math.py`) needs no change: rope/apply_rope already fp32, and `jax.nn.dot_product_attention` upcasts logits+softmax to fp32 internally (verified for jax 0.10.2).

- [ ] **Step 4: Run new + existing flux1 tests**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/models/flux1/ -v`
Expected: all PASS (existing tests may need their expected dtypes updated from bf16 → fp32 — that change is part of this task).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/models/flux1/ tests/models/flux1/
git commit -m "feat: mixed-precision Flux1 (fp32 master weights, bf16 compute knob, fp32 islands)"
```

---

### Task 5: Flux1Joint

**Files:**
- Modify: `src/gensbi/models/flux1joint/model.py` (Flux1JointParams line ~95: `param_dtype` → fp32, add `dtype=jnp.bfloat16`; thread `dtype=params.dtype` into every constructor; learned tokens at lines ~187/193 stay at `param_dtype`; input casts at lines ~231-232 → `jnp.float32`)
- Test: `tests/models/flux1joint/test_flux1joint_precision.py`

**Interfaces:**
- Consumes: Task 4's updated `flux1/layers.py` signatures (`dtype=..., param_dtype=...`) — flux1joint imports its blocks from there.
- Produces: `Flux1JointParams(param_dtype=jnp.float32, dtype=jnp.bfloat16)`; `Flux1Joint` output fp32.

- [ ] **Step 1: Write failing tests** — copy `tests/models/flux1joint/`'s existing smallest model fixture and add the same four tests as Task 4 (`test_master_weights_fp32_with_bf16_compute`, `test_output_is_fp32`, `test_grads_are_fp32`, `test_bf16_close_to_fp32` with rel-tolerance 2e-2), adapted to `Flux1Joint`'s call signature (it takes `obs`, `t`, `condition_mask` — mirror the existing tests' invocation exactly).
- [ ] **Step 2: Run** `... -m pytest tests/models/flux1joint/test_flux1joint_precision.py -v` — Expected: FAIL, no field `dtype`.
- [ ] **Step 3: Implement** — same pattern as Task 4: params fields, thread `dtype`/`param_dtype` into every layer constructor, `LastLayer` already emits fp32 via Task 4, input casts → fp32. Learned `nnx.Param` tokens keep `dtype=params.param_dtype` (they are master weights).
- [ ] **Step 4: Run** `... -m pytest tests/models/flux1joint/ -v` — Expected: all PASS (update stale dtype expectations in existing tests if any).
- [ ] **Step 5: Commit** `git add src/gensbi/models/flux1joint/ tests/models/flux1joint/ && git commit -m "feat: mixed-precision Flux1Joint"`

---

### Task 6: Simformer

**Files:**
- Modify: `src/gensbi/models/simformer/model.py` (SimformerParams line ~77: add `dtype: DTypeLike = jnp.bfloat16`; thread; input casts lines ~190-191 → fp32; output projection pinned fp32)
- Modify: `src/gensbi/models/simformer/transformer.py` (AttentionBlock, DenseBlock, Transformer)
- Test: `tests/models/simformer/test_simformer_precision.py`

**Interfaces:**
- Produces: `SimformerParams(param_dtype=jnp.float32, dtype=jnp.bfloat16)`; Simformer output fp32.

**Design note (fp32 softmax island):** `flax nnx.MultiHeadAttention` computes softmax in its `dtype` (no fp32 forcing in flax 0.12.7). Rather than a custom `attention_fn`, keep the whole `AttentionBlock` at `dtype=jnp.float32` — a slightly larger island. The FLOPs are dominated by the `DenseBlock` MLP stack (widening_factor 4), which does get bf16. Record this in the AttentionBlock docstring.

- [ ] **Step 1: Write failing tests** — same four tests as Task 4, using the smallest `SimformerParams` construction found in `tests/models/simformer/` (copy its fixture; add `dtype=...` argument). Assert output fp32, params fp32, grads fp32, bf16-vs-fp32 rel-err < 2e-2.
- [ ] **Step 2: Run** `... -m pytest tests/models/simformer/test_simformer_precision.py -v` — Expected: FAIL, no field `dtype`.
- [ ] **Step 3: Implement**
  - `SimformerParams`: add `dtype: DTypeLike = jnp.bfloat16` (param_dtype already fp32).
  - `transformer.py`: add `dtype: DTypeLike = jnp.float32` keyword to AttentionBlock/DenseBlock/Transformer. AttentionBlock: `nnx.MultiHeadAttention(..., dtype=jnp.float32, ...)` and its LayerNorm `dtype=jnp.float32` (fp32 island — ignore the passed knob for the attention itself, but accept and store it for future use is NOT needed; simply construct MHA fp32). DenseBlock: LayerNorm `dtype=jnp.float32`; all `nnx.Linear` including `context_block` get `dtype=dtype`. Transformer: forward `dtype` to blocks; final `self.layer_norm` gets `dtype=jnp.float32`.
  - `model.py`: thread `dtype=params.dtype` into the Transformer, token/value embedders and any `nnx.Linear`; the LAST Linear producing the output gets `dtype=jnp.float32` (emit-fp32 contract); input casts → `jnp.float32`. The learned `nnx.Param` at line ~130 stays at `param_dtype`.
- [ ] **Step 4: Run** `... -m pytest tests/models/simformer/ -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `git add src/gensbi/models/simformer/ tests/models/simformer/ && git commit -m "feat: mixed-precision Simformer (bf16 MLP compute, fp32 attention island)"`

---

### Task 7: MAF — dtype knob, fp32 default, bit-identical

**Files:**
- Modify: `src/gensbi/models/maf/model.py` (MAFlowParams: add `param_dtype: DTypeLike = jnp.float32` and `dtype: DTypeLike = jnp.float32` fields; thread into MADE construction)
- Modify: `src/gensbi/models/maf/made.py`, `src/gensbi/models/maf/masked_linear.py` (add `dtype` kwarg, default `jnp.float32`, forwarded to `nnx.Linear`-equivalents)
- Test: `tests/models/maf/test_maf_precision.py`

**Interfaces:**
- Produces: `MAFlowParams(param_dtype=jnp.float32, dtype=jnp.float32)` — knob present, fp32 default. Log-det accumulation paths remain unconditionally fp32 (no `dtype` applied to log-det sums).

- [ ] **Step 1: Capture the bit-identical baseline BEFORE any edit** — write the test against CURRENT code and check it passes, so the refactor is provably a no-op:

```python
# tests/models/maf/test_maf_precision.py
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import MAFlow, MAFlowParams
from tests.precision_utils import assert_tree_dtype


def _make(**kw):
    return MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, **kw))


def test_params_fp32():
    m = _make()
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)


def test_log_prob_fp32_and_finite():
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    assert lp.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lp))


def test_refactor_is_bit_identical():
    # Golden values computed from the pre-refactor code at the start of this
    # task; regenerate with the command in the plan and paste here.
    m = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    cond = jnp.ones((5, 2))
    lp = m.log_prob(x, cond)
    expected = jnp.asarray(GOLDEN_LOG_PROBS)  # paste from Step 1 output
    assert jnp.array_equal(lp, expected), "MAF refactor must be a bit-exact no-op"
```

Generate `GOLDEN_LOG_PROBS` on unmodified code with:
`JAX_PLATFORMS=cpu .../python -c "import jax, jax.numpy as jnp; from flax import nnx; from gensbi.models import MAFlow, MAFlowParams; m=MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2)); x=jax.random.normal(jax.random.PRNGKey(1),(5,3)); print(repr(m.log_prob(x, jnp.ones((5,2)))))"`
and paste the printed list into the test file as `GOLDEN_LOG_PROBS = [...]`. Adjust `log_prob`'s exact call signature to match `MAFlow`'s API (check `src/gensbi/models/maf/model.py`).

- [ ] **Step 2: Run on unmodified code** — all three tests must PASS before the refactor (this task's "failing test" is the `dtype=` kwarg not existing yet; verify: constructing `MAFlowParams(..., dtype=jnp.float32)` raises `TypeError`).
- [ ] **Step 3: Implement** — add the two fields to `MAFlowParams`; add `dtype: DTypeLike = jnp.float32` kwargs to MADE/MaskedLinear, forwarded to their linear layers; thread `params.dtype`/`params.param_dtype` from `MAFlow.__init__`. Do NOT touch log-det arithmetic.
- [ ] **Step 4: Run** `... -m pytest tests/models/maf/ tests/normalizing_flows/ -v` — Expected: all PASS including the bit-identical golden test.
- [ ] **Step 5: Commit** `git add src/gensbi/models/maf/ tests/models/maf/ && git commit -m "feat: dtype knob for MAF (fp32 default, bit-identical refactor)"`

---

### Task 8: TarFlow — dtype knob, fp32 default, bit-identical

**Files:**
- Modify: `src/gensbi/models/tarflow/model.py` (TarFlowParams: add `param_dtype: DTypeLike = jnp.float32`, `dtype: DTypeLike = jnp.float32`; thread)
- Modify: `src/gensbi/models/tarflow/blocks.py` (AttentionBlock/MetaBlock and any Linear: accept + forward `dtype`, `param_dtype`)
- Modify: `src/gensbi/models/tarflow/conditioners.py`, `src/gensbi/models/core/tokenizers.py` (or wherever the tokenizer Linears live — locate with `grep -rn "nnx.Linear" src/gensbi/models/tarflow src/gensbi/models/core`): same threading
- Test: `tests/models/tarflow/test_tarflow_precision.py`

**Interfaces:**
- Produces: `TarFlowParams(param_dtype=jnp.float32, dtype=jnp.float32)`. Hard-fp32 regardless of knob: the softplus/soft_clip affine-scale path in MetaBlock (already fp32 — keep), log-det sums, `mean`/`std` standardization buffers.

- [ ] **Step 1: Golden baseline + tests** — same structure as Task 7: `test_params_fp32`, `test_log_prob_fp32_and_finite`, `test_refactor_is_bit_identical` with `GOLDEN_LOG_PROBS` generated from unmodified code using the smallest `TarFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, num_blocks=2, layers_per_block=1)` vector-mode model (mirror the fixtures in `tests/models/tarflow/`). Also verify `TarFlowParams(..., dtype=jnp.float32)` currently raises `TypeError`.
- [ ] **Step 2: Run on unmodified code** — golden tests PASS; kwarg test FAILS as expected.
- [ ] **Step 3: Implement** — add fields; thread through blocks/conditioners/tokenizers. The KV-cache buffers in sampling (`blocks.py` line ~438) and the softplus/soft_clip scale computation keep their current dtype behavior untouched.
- [ ] **Step 4: Run** `... -m pytest tests/models/tarflow/ -v` — Expected: all PASS including bit-identical golden and the 18-test cached≡reference sampling gate.
- [ ] **Step 5: Commit** `git add src/gensbi/models/tarflow/ src/gensbi/models/core/ tests/models/tarflow/ && git commit -m "feat: dtype knob for TarFlow (fp32 default, bit-identical refactor)"`

---

### Task 9: PixelDiT

**Files:**
- Modify: `src/gensbi/experimental/models/pixeldit/model.py` (PixelDiTParams line ~71: `param_dtype` bf16 → fp32, add `dtype: DTypeLike = jnp.bfloat16`; thread; input casts → fp32)
- Modify: `src/gensbi/experimental/models/pixeldit/embedders.py`, `modules.py`, `blocks.py` (signature defaults `param_dtype` bf16 → fp32; add `dtype=jnp.bfloat16`; thread — compute Linears/convs get `dtype`, all norm layers get `dtype=jnp.float32`, the final output projection gets `dtype=jnp.float32`; `rope.py` is already fp32 math, leave it)
- Test: `tests/experimental/pixeldit/test_pixeldit_precision.py` (place beside the existing pixeldit tests — locate with `ls tests/experimental/`)

**Interfaces:**
- Produces: `PixelDiTParams(param_dtype=jnp.float32, dtype=jnp.bfloat16)`; PixelDiT output fp32.

- [ ] **Step 1: Write failing tests** — the same four tests as Task 4 (params-fp32, output-fp32, grads-fp32, bf16-close-to-fp32 rel < 2e-2), built on the smallest PixelDiT fixture from the existing experimental tests.
- [ ] **Step 2: Run** — Expected: FAIL, no field `dtype`.
- [ ] **Step 3: Implement** — canonical pattern; timestep embedding path pinned fp32 (cast `t` to fp32 at the model door, embed in fp32, let the first Linear downcast); zero-init output projection = fp32 island.
- [ ] **Step 4: Run** `... -m pytest tests/experimental/ -k pixeldit -v` — Expected: all PASS (update stale bf16 dtype expectations in existing tests as needed).
- [ ] **Step 5: Commit** `git add src/gensbi/experimental/models/pixeldit/ tests/experimental/ && git commit -m "feat: mixed-precision PixelDiT (fp32 master weights, bf16 compute)"`

---

### Task 10: FieldDiT

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (FieldDiTParams line ~72: `param_dtype` bf16 → fp32, add `dtype: DTypeLike = jnp.bfloat16`; input casts lines ~195-196 → fp32; timestep embedding line ~216: `.astype(self.param_dtype)` → `.astype(self.dtype)`)
- Modify: `src/gensbi/experimental/models/fielddit/blocks.py`, `codec.py`, `cond.py`, `core.py` (thread `dtype`; GroupNorms and LayerNorms `dtype=jnp.float32`; zero-init `conv_out` final projection `dtype=jnp.float32`)
- Test: `tests/experimental/fielddit/test_fielddit_precision.py` (beside existing fielddit tests)

**Interfaces:**
- Produces: `FieldDiTParams(param_dtype=jnp.float32, dtype=jnp.bfloat16)`; FieldDiT output fp32.

- [ ] **Step 1: Write failing tests** — same four tests as Task 4 on the smallest FieldDiT fixture from the existing fielddit tests (small H=W, hidden size — copy the existing conftest fixture).
- [ ] **Step 2: Run** — Expected: FAIL, no field `dtype`.
- [ ] **Step 3: Implement** — canonical pattern across the four module files; MMDiT core blocks come from `flux1.layers` (already threaded in Task 4 — pass `dtype=p.dtype` where those are constructed).
- [ ] **Step 4: Run** `... -m pytest tests/experimental/ -k fielddit -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `git add src/gensbi/experimental/models/fielddit/ tests/experimental/ && git commit -m "feat: mixed-precision FieldDiT"`

---

### Task 11: Autoencoders

**Files:**
- Modify: `src/gensbi/experimental/models/autoencoders/commons.py` (AutoEncoderParams: `param_dtype` bf16 → fp32, add `dtype: DTypeLike = jnp.bfloat16`)
- Modify: `src/gensbi/experimental/models/autoencoders/autoencoder_1d.py`, `autoencoder_2d.py` (signature defaults flipped; thread `dtype`; GroupNorms `dtype=jnp.float32`; final decoder conv `dtype=jnp.float32`)
- Modify: `src/gensbi/experimental/recipes/vae_pipeline.py` (only if it constructs layers with explicit dtypes — check with `grep -n dtype`; its loss reduction gets the fp32 cast pattern from Task 2)
- Test: `tests/experimental/autoencoders/test_autoencoder_precision.py`

**Interfaces:**
- Produces: `AutoEncoderParams(param_dtype=jnp.float32, dtype=jnp.bfloat16)`; encoder/decoder outputs fp32.

- [ ] **Step 1: Write failing tests** — params-fp32 + output-fp32 + grads-fp32 for the 2D autoencoder's smallest existing fixture (closeness test optional here; include if a same-seed fp32/bf16 pair is easy with the existing fixture).
- [ ] **Step 2: Run** — Expected: FAIL, no field `dtype`.
- [ ] **Step 3: Implement** — canonical pattern; `DiagonalGaussian` sampling stays fp32.
- [ ] **Step 4: Run** `... -m pytest tests/experimental/ -k autoencoder -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `git add src/gensbi/experimental/models/autoencoders/ src/gensbi/experimental/recipes/ tests/experimental/ && git commit -m "feat: mixed-precision autoencoders"`

---

### Task 12: Pipeline guard, EMA regression test, orbax/safetensors restore

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py` (add `_warn_if_not_fp32_master_weights(model)` called from `Pipeline.__init__` after `self.model = model`; and in `restore_model` (line ~517) cast restored leaves to the target model's dtypes before `nnx.merge` — mirror the `arr.astype(want.dtype)` approach already used in `load_safetensors`)
- Test: `tests/recipes/test_precision_pipeline.py`, `tests/utils/test_serialization_dtype.py`

**Interfaces:**
- Consumes: fp32-defaulted models from Tasks 4-11.
- Produces: `gensbi.recipes.pipeline._warn_if_not_fp32_master_weights(model)` — emits `UserWarning` naming offending leaves when any floating `nnx.Param` is not fp32; no-op otherwise.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recipes/test_precision_pipeline.py
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from gensbi.recipes.pipeline import _warn_if_not_fp32_master_weights


def test_warns_on_bf16_master_weights():
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    with pytest.warns(UserWarning, match="fp32"):
        _warn_if_not_fp32_master_weights(model)


def test_silent_on_fp32_master_weights(recwarn):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    _warn_if_not_fp32_master_weights(model)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_ema_integrates_small_updates_in_fp32():
    # The bug-shaped test: decay=0.999 EMA must integrate 0.1%-scale updates.
    decay = 0.999
    tx = optax.ema(decay)
    w = jnp.ones((64,), jnp.float32)
    state = tx.init(w)
    for _ in range(500):
        w = w * 1.001
        _, state = tx.update(w, state)
    # fp32 reference computed in float64
    import numpy as np
    w64, ema64 = np.ones(1), np.ones(1)
    for _ in range(500):
        w64 = w64 * 1.001
        ema64 = decay * ema64 + (1 - decay) * w64
    # optax debiases; compare the raw accumulator
    assert abs(float(state.ema[0]) - float(ema64[0])) < 1e-3


def test_adamw_moments_are_fp32_for_fp32_params():
    # Spec 5(d): optimizer state must be fp32 once master weights are fp32.
    from tests.precision_utils import assert_tree_dtype
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.float32,
                       dtype=jnp.bfloat16)
    opt = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    mu = jax.tree.map(lambda x: x, opt.opt_state)  # traverse whole opt state
    leaves = [l for l in jax.tree.leaves(mu)
              if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
    assert leaves and all(l.dtype == jnp.float32 for l in leaves)


def test_ema_bf16_demonstrates_the_old_bug():
    # Documents WHY master weights must be fp32: the same accumulation in
    # bf16 diverges badly (increment below mantissa resolution + rounded decay).
    decay = 0.999
    tx = optax.ema(decay)
    w = jnp.ones((64,), jnp.bfloat16)
    state = tx.init(w)
    for _ in range(500):
        w = (w.astype(jnp.float32) * 1.001).astype(jnp.bfloat16)
        _, state = tx.update(w, state)
    import numpy as np
    w64, ema64 = np.ones(1), np.ones(1)
    for _ in range(500):
        w64 = w64 * 1.001
        ema64 = decay * ema64 + (1 - decay) * w64
    err = abs(float(state.ema.astype(jnp.float32)[0]) - float(ema64[0]))
    assert err > 1e-2, "bf16 EMA unexpectedly accurate — did optax change?"
```

```python
# tests/utils/test_serialization_dtype.py
import jax.numpy as jnp
from flax import nnx

from gensbi.utils.serialization import save_safetensors, load_safetensors


def test_bf16_checkpoint_loads_into_fp32_model(tmp_path):
    # Old checkpoints (bf16 master weights) must load into new fp32 models.
    old = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    p = tmp_path / "old.safetensors"
    save_safetensors(old, p)
    new = nnx.Linear(2, 3, rngs=nnx.Rngs(1), param_dtype=jnp.float32)
    load_safetensors(new, p)
    assert new.kernel[...].dtype == jnp.float32
    assert jnp.allclose(
        new.kernel[...], old.kernel[...].astype(jnp.float32)
    )
```

- [ ] **Step 2: Run to verify failure** — the guard tests fail with `ImportError` (`_warn_if_not_fp32_master_weights` doesn't exist); the serialization test may already PASS (`load_safetensors` already casts via `arr.astype(want.dtype)`) — keep it as a regression lock either way; EMA tests should PASS immediately (they test optax behavior, documenting the fix rationale).

- [ ] **Step 3: Implement the guard + orbax cast**

```python
# in src/gensbi/recipes/pipeline.py (module level, near ModelEMA)
import warnings
import flax.traverse_util as tu


def _warn_if_not_fp32_master_weights(model):
    """Warn when trainable params are not fp32 master weights.

    Mixed precision in GenSBI stores master weights in fp32 and selects the
    compute dtype via each model's ``dtype`` knob; bf16 master weights break
    AdamW moment accumulation and make optax.ema unable to integrate
    (1 - decay)-scale updates.
    """
    flat = tu.flatten_dict(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    bad = {
        ".".join(str(p) for p in k): str(v.dtype)
        for k, v in flat.items()
        if jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != jnp.float32
    }
    if bad:
        warnings.warn(
            "model has non-fp32 master weights (training will be numerically "
            f"degraded; set param_dtype=jnp.float32 and use the dtype knob "
            f"for compute instead): {bad}",
            UserWarning,
            stacklevel=3,
        )
```

Call it in `Pipeline.__init__` right after the model is stored. In `restore_model` (line ~517-556), after `read_mgr.restore(...)` and before `nnx.merge`, map the restored state leaves through `.astype` of the corresponding target-model leaf dtype (same loop shape as `load_safetensors`); apply to both the model and EMA restore paths.

- [ ] **Step 4: Run** `... -m pytest tests/recipes/test_precision_pipeline.py tests/utils/test_serialization_dtype.py tests/recipes/ -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `git add src/gensbi/recipes/pipeline.py tests/recipes/ tests/utils/ && git commit -m "feat: fp32 master-weight guard, EMA regression tests, dtype-safe checkpoint restore"`

---

### Task 13: Full-suite regression and wrap-up

**Files:**
- Modify: none expected (fixes only if the sweep finds stragglers)

- [ ] **Step 1: Sweep for leftover bf16 param defaults**

Run: `grep -rn "param_dtype: DTypeLike = jnp.bfloat16\|param_dtype=jnp.bfloat16" src/gensbi/`
Expected: no hits (any hit is a missed constructor — fix it with the canonical pattern and add it to the owning task's test).

- [ ] **Step 2: Run the entire test suite**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/ -q`
Expected: all PASS, no new skips.

- [ ] **Step 3: Smoke the recovery scripts (CPU, smoke mode)**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python scripts/maf_nle_recovery.py --smoke` (and the tarflow twin; check `scripts/` for exact names/flags)
Expected: completes without NaN/dtype errors.

- [ ] **Step 4: Commit any straggler fixes**

```bash
git add -A && git commit -m "fix: mixed-precision stragglers from full-suite sweep"
```

- [ ] **Step 5: Hand off the GPU validation gate to the user**

Not automatable here — report to the user that the branch is ready for:
1. PixelDiT GRF probe rerun with `use_ema=True` (expect structure, not white noise).
2. Flux1 two-moons sanity run (mixed-precision convergence ≥ current).
