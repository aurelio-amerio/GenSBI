# TarFlow (TransformerFlow) v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a transformer autoregressive normalizing flow (`TransformerFlow`, adapted from TarFlow) to `gensbi.normalizing_flows`, exposing exact `log_prob` + a sequential sampler so it serves NLE (via the existing `NLEPosterior`) and vector→vector NPE (via the existing `ConditionalFlowPipeline`).

**Architecture:** A batched-native `(B, T, F)` density model, sibling to the per-example `Flow`/`Chain` MAF track (NOT built on `Chain`). Two swappable seams — an invertible `VectorTokenizer` for the modeled variable and a per-block `VectorConditioner` for θ — feed a stack of `MetaBlock`s (causal-masked attention → per-token affine → shift-by-one → triangular Jacobian). Exposes the same duck-typed surface (`log_prob`, `sample`, `set_standardization`) so the existing pipeline and NLE wrapper consume it unchanged.

**Tech Stack:** JAX, Flax NNX, `jax.nn.dot_product_attention`, NumPyro (prior/NUTS, via existing `NLEPosterior`), grain (datasets in tests), pytest.

Design spec: `docs/superpowers/specs/2026-06-23-tarflow-nle-design.md`. Reference: `reference/ml-tarflow/transformer_flow.py`.

## Global Constraints

- **Precision:** float32 throughout (matches the existing NF track). Accumulate log-dets in float32.
- **Run tests with:** `JAX_PLATFORMS=cpu .venv/bin/python -m pytest` (GPUs are usually busy; `cpu` is also what the existing NF tests pin).
- **Module location:** `src/gensbi/normalizing_flows/transformer_flow/` (production NF track).
- **NNX conventions:** learnable tensors are `nnx.Param`; fixed buffers (permutations, standardization mean/std) are `gensbi.normalizing_flows.bijections.base.Mask` (an `nnx.Variable` subclass — excluded from `nnx.split(wrt=nnx.Param)`/EMA, still checkpointed).
- **Direction convention (match `Bijection`):** `inverse` = data→noise (density; the fast parallel pass); `forward` = noise→data (sampling; the sequential pass).
- **Causal attention:** call `jax.nn.dot_product_attention(q, k, v, is_causal=True)` directly (XLA, fp32). We do NOT use the Flux1 `attention()` wrapper in v1 — it is rope/`pe`-oriented and does not expose `is_causal`; reuse is at the `dot_product_attention` primitive level. (The shared wrapper + rope2d + cuDNN-flash is the v2 path.)
- **Exactness invariants (must not regress):** (1) token *i*'s affine params depend only on tokens `< i` (causal mask + shift-by-one) ⇒ `log|det| = −Σa`; (2) normalize over the **channel** axis only (`nnx.LayerNorm` over the last dim), never across tokens; (3) the tokenizer is a volume-preserving reshape (log-det 0); (4) `zero_init` ⇒ each block is identity ⇒ the flow starts at the standardized-Gaussian base.
- **Attribution:** every new source file starts with a docstring line: `Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.` The license file already exists at `src/gensbi/normalizing_flows/transformer_flow/LICENSE.apple`.

---

### Task 1: VectorTokenizer (invertible reshape seam)

**Files:**
- Create: `src/gensbi/normalizing_flows/transformer_flow/__init__.py` (empty for now)
- Create: `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py`
- Test: `tests/normalizing_flows/transformer_flow/__init__.py` (empty)
- Test: `tests/normalizing_flows/transformer_flow/test_tokenizers.py`

**Interfaces:**
- Produces: `VectorTokenizer(dim: int, block_size: int = 1)` with attributes `.dim`, `.T`, `.F`; methods `tokenize(x: (B, dim)) -> (B, T, F)` and `detokenize(tokens: (B, T, F)) -> (B, dim)`. Plain class (no params). `T = dim // block_size`, `F = block_size`.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_tokenizers.py
import jax.numpy as jnp
import pytest
from gensbi.normalizing_flows.transformer_flow.tokenizers import VectorTokenizer


def test_shapes_scalar_per_token():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert (tok.T, tok.F) == (6, 1)
    x = jnp.arange(12.0).reshape(2, 6)
    t = tok.tokenize(x)
    assert t.shape == (2, 6, 1)


def test_shapes_block_per_token():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert (tok.T, tok.F) == (3, 2)
    x = jnp.arange(12.0).reshape(2, 6)
    assert tok.tokenize(x).shape == (2, 3, 2)


def test_roundtrip_identity():
    tok = VectorTokenizer(dim=6, block_size=2)
    x = jnp.arange(12.0).reshape(2, 6)
    assert jnp.allclose(tok.detokenize(tok.tokenize(x)), x)


def test_block_size_must_divide_dim():
    with pytest.raises(ValueError):
        VectorTokenizer(dim=5, block_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_tokenizers.py -v`
Expected: FAIL (`ModuleNotFoundError` / `No module named ...transformer_flow.tokenizers`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/normalizing_flows/transformer_flow/tokenizers.py
"""Invertible tokenizers for the transformer flow (the modeled-variable seam).

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

A tokenizer maps the modeled variable to a token sequence ``(B, T, F)`` and back.
It MUST be volume-preserving (a fixed invertible reshape, log-det 0) — never a
learned lossy encoder — so the change-of-variables stays exact.
"""

from jax import Array


class VectorTokenizer:
    """Reshape a vector ``(B, dim)`` into ``T = dim // block_size`` tokens of
    ``F = block_size`` features. Pure reshape: invertible, log-det 0."""

    def __init__(self, dim: int, block_size: int = 1):
        if dim % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide dim ({dim})")
        self.dim = dim
        self.F = block_size
        self.T = dim // block_size

    def tokenize(self, x: Array) -> Array:
        return x.reshape(x.shape[0], self.T, self.F)

    def detokenize(self, tokens: Array) -> Array:
        return tokens.reshape(tokens.shape[0], self.dim)
```

Also create the two empty `__init__.py` files (`src/.../transformer_flow/__init__.py` and `tests/normalizing_flows/transformer_flow/__init__.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_tokenizers.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/__init__.py \
        src/gensbi/normalizing_flows/transformer_flow/tokenizers.py \
        tests/normalizing_flows/transformer_flow/
git commit -m "feat(nf): VectorTokenizer (invertible reshape seam) for transformer flow"
```

---

### Task 2: VectorConditioner (per-token-add seam)

**Files:**
- Create: `src/gensbi/normalizing_flows/transformer_flow/conditioners.py`
- Test: `tests/normalizing_flows/transformer_flow/test_conditioners.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `VectorConditioner(cond_dim: int, channels: int, rngs: nnx.Rngs)` (an `nnx.Module`) with:
  - `embed(cond: (B, cond_dim) | None) -> (B, channels) | None` — returns `None` if `cond_dim == 0`; raises `ValueError` if `cond_dim > 0` and `cond is None`.
  - `inject(tokens: (B, T, channels), signal: (B, channels) | None) -> (B, T, channels)` — broadcast-adds the signal to every token; pass-through if `signal is None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_conditioners.py
import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner


def test_embed_shape_and_inject_broadcasts():
    cond_dim, channels, B, T = 3, 8, 4, 5
    c = VectorConditioner(cond_dim, channels, rngs=nnx.Rngs(0))
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim))
    sig = c.embed(cond)
    assert sig.shape == (B, channels)
    tokens = jnp.zeros((B, T, channels))
    out = c.inject(tokens, sig)
    assert out.shape == (B, T, channels)
    # same signal added to every token
    assert jnp.allclose(out[:, 0, :], sig)
    assert jnp.allclose(out[:, 0, :], out[:, T - 1, :])


def test_unconditional_passthrough():
    c = VectorConditioner(0, 8, rngs=nnx.Rngs(0))
    assert c.embed(jnp.zeros((4, 0))) is None
    tokens = jnp.ones((4, 5, 8))
    assert jnp.allclose(c.inject(tokens, None), tokens)


def test_missing_cond_raises():
    c = VectorConditioner(3, 8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        c.embed(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/normalizing_flows/transformer_flow/conditioners.py
"""Conditioning seams for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

v1 ``VectorConditioner`` is the continuous analog of TarFlow's ``class_embed``:
an MLP embeds the condition to a ``channels``-vector that is broadcast-added to
every token. The signal depends only on the condition (constant w.r.t. the
modeled variable), so it shifts the affine params without breaking the
triangular Jacobian. A plain 2-layer MLP is used (not ``MLPEmbedder``, whose
``hidden_dim % in_dim == 0`` constraint does not fit arbitrary ``cond_dim``).
"""

import jax
from flax import nnx
from jax import Array


class VectorConditioner(nnx.Module):
    """MLP(cond) → per-token additive bias. ``cond_dim == 0`` ⇒ unconditional."""

    def __init__(self, cond_dim: int, channels: int, rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        self.channels = channels
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond: Array | None) -> Array | None:
        if self.cond_dim == 0:
            return None
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        return self.l2(jax.nn.silu(self.l1(cond)))

    def inject(self, tokens: Array, signal: Array | None) -> Array:
        if signal is None:
            return tokens
        return tokens + signal[:, None, :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/conditioners.py \
        tests/normalizing_flows/transformer_flow/test_conditioners.py
git commit -m "feat(nf): VectorConditioner (per-token-add) for transformer flow"
```

---

### Task 3: AttentionBlock (causal transformer block)

**Files:**
- Create: `src/gensbi/normalizing_flows/transformer_flow/blocks.py`
- Test: `tests/normalizing_flows/transformer_flow/test_blocks_attention.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AttentionBlock(channels: int, head_dim: int, expansion: int, rngs: nnx.Rngs)` (an `nnx.Module`) with `__call__(x: (B, T, channels)) -> (B, T, channels)`. Pre-norm residual; self-attention is **causal** (`is_causal=True`). Requires `channels % head_dim == 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_blocks_attention.py
import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.normalizing_flows.transformer_flow.blocks import AttentionBlock


def test_output_shape():
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    assert blk(x).shape == (2, 5, 8)


def test_attention_is_causal():
    """output[i] must not depend on input[j] for j > i (causal mask)."""
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    T, C = 4, 8
    x0 = jax.random.normal(jax.random.PRNGKey(2), (1, T, C))

    def f(x):
        return blk(x[None])[0]            # (T, C)

    J = jax.jacrev(f)(x0[0])              # (T, C, T, C)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_attention.py -v`
Expected: FAIL (`ImportError: cannot import name 'AttentionBlock'`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/normalizing_flows/transformer_flow/blocks.py
"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


class AttentionBlock(nnx.Module):
    """Pre-norm residual block: causal self-attention + MLP.

    LayerNorm is over the channel axis only (never across tokens), so it does
    not leak future tokens into earlier ones.
    """

    def __init__(self, channels: int, head_dim: int, expansion: int,
                 rngs: nnx.Rngs):
        if channels % head_dim != 0:
            raise ValueError(
                f"channels ({channels}) must be a multiple of head_dim "
                f"({head_dim})")
        self.num_heads = channels // head_dim
        self.head_dim = head_dim
        self.norm1 = nnx.LayerNorm(channels, rngs=rngs)
        self.qkv = nnx.Linear(channels, 3 * channels, rngs=rngs)
        self.proj = nnx.Linear(channels, channels, rngs=rngs)
        self.norm2 = nnx.LayerNorm(channels, rngs=rngs)
        self.mlp_in = nnx.Linear(channels, channels * expansion, rngs=rngs)
        self.mlp_out = nnx.Linear(channels * expansion, channels, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, hd)
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.reshape(B, T, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_attention.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py \
        tests/normalizing_flows/transformer_flow/test_blocks_attention.py
git commit -m "feat(nf): causal AttentionBlock for transformer flow"
```

---

### Task 4: MetaBlock (one exact bijection over tokens)

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (append `MetaBlock`)
- Test: `tests/normalizing_flows/transformer_flow/test_blocks_meta.py`

**Interfaces:**
- Consumes: `AttentionBlock` (Task 3); `VectorConditioner` (Task 2); `Mask` from `gensbi.normalizing_flows.bijections.base`.
- Produces: `MetaBlock(F, channels, T, perm, inv_perm, conditioner, num_layers, head_dim, expansion, rngs, zero_init=True)` (an `nnx.Module`) with:
  - `inverse(x: (B, T, F), cond=None) -> (z: (B, T, F), logdet: (B,))` — data→noise, parallel single pass; `logdet = −Σ a`.
  - `forward(z: (B, T, F), cond=None) -> (x: (B, T, F), logdet: (B,))` — noise→data, sequential `lax.scan`; `logdet = +Σ a`.
  - `perm`/`inv_perm` are length-`T` int arrays stored as `Mask` buffers.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_blocks_meta.py
import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.normalizing_flows.transformer_flow.blocks import MetaBlock
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner


def _make(T=4, F=1, channels=8, cond_dim=2, zero_init=True, rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)                     # identity perm
    inv_perm = jnp.argsort(perm)
    cond = VectorConditioner(cond_dim, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm, inv_perm=inv_perm,
                     conditioner=cond, num_layers=2, head_dim=4, expansion=2,
                     rngs=rngs, zero_init=zero_init)


def test_zero_init_is_identity():
    blk = _make(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (3, 2))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)


def test_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i (F=1 ⇒ clean (T,T) Jacobian)."""
    blk = _make(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])     # (1, T, 1)
        return z[0, :, 0]                            # (T,)

    J = jax.jacrev(f)(x0[:, 0])                      # (T, T)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_inverse_logdet_matches_autodiff():
    blk = _make(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_forward_inverse_roundtrip():
    blk = _make(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_meta.py -v`
Expected: FAIL (`ImportError: cannot import name 'MetaBlock'`).

- [ ] **Step 3: Write minimal implementation**

Append to `src/gensbi/normalizing_flows/transformer_flow/blocks.py`:

```python
from gensbi.normalizing_flows.bijections.base import Mask


class MetaBlock(nnx.Module):
    """One exact autoregressive bijection over a token sequence.

    inverse (data→noise): per-token affine ``z = (x − b)·exp(−a)`` with
    ``(a, b)`` produced by causal attention + shift-by-one, so token i's params
    depend only on tokens < i ⇒ triangular Jacobian, ``logdet = −Σ a``.
    forward (noise→data): sequential scan re-running the causal pass on the
    partially-built sequence (mirrors ``MaskedAutoregressive.forward``).
    """

    def __init__(self, F, channels, T, perm, inv_perm, conditioner,
                 num_layers, head_dim, expansion, rngs, zero_init=True):
        self.F = F
        self.T = T
        self.perm = Mask(jnp.asarray(perm, dtype=jnp.int32))
        self.inv_perm = Mask(jnp.asarray(inv_perm, dtype=jnp.int32))
        self.conditioner = conditioner
        self.proj_in = nnx.Linear(F, channels, rngs=rngs)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (T, channels)) * 1e-2)
        self.attn_blocks = nnx.List(
            [AttentionBlock(channels, head_dim, expansion, rngs)
             for _ in range(num_layers)])
        self.proj_out = nnx.Linear(channels, 2 * F, rngs=rngs)
        if zero_init:
            self.proj_out.kernel[...] = jnp.zeros_like(self.proj_out.kernel[...])
            self.proj_out.bias[...] = jnp.zeros_like(self.proj_out.bias[...])

    def _params(self, x_perm: Array, cond: Array | None):
        """(a, b) for the permuted tokens; shifted so params[i] sees tokens < i."""
        signal = self.conditioner.embed(cond)
        h = self.proj_in(x_perm) + self.pos_embed[...][None]   # (B, T, C)
        h = self.conditioner.inject(h, signal)
        for blk in self.attn_blocks:
            h = blk(h)
        out = self.proj_out(h)                                 # (B, T, 2F)
        out = jnp.concatenate(
            [jnp.zeros_like(out[:, :1]), out[:, :-1]], axis=1)  # shift-by-one
        a, b = jnp.split(out, 2, axis=-1)                      # each (B, T, F)
        return a, b

    def inverse(self, x: Array, cond: Array | None = None):
        xp = x[:, self.perm[...], :]
        a, b = self._params(xp, cond)
        z = (xp - b) * jnp.exp(-a)
        logdet = -jnp.sum(a, axis=(1, 2))                      # (B,)
        z = z[:, self.inv_perm[...], :]
        return z, logdet

    def forward(self, z: Array, cond: Array | None = None):
        zp = z[:, self.perm[...], :]

        def body(x, i):
            a, b = self._params(x, cond)        # a[:,i],b[:,i] depend on tokens < i
            xi = zp[:, i, :] * jnp.exp(a[:, i, :]) + b[:, i, :]
            return x.at[:, i, :].set(xi), None

        x = jnp.zeros_like(zp)
        x, _ = jax.lax.scan(body, x, jnp.arange(self.T))
        a, _ = self._params(x, cond)
        logdet = jnp.sum(a, axis=(1, 2))                       # (B,), +Σa
        x = x[:, self.inv_perm[...], :]
        return x, logdet
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_meta.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py \
        tests/normalizing_flows/transformer_flow/test_blocks_meta.py
git commit -m "feat(nf): MetaBlock (exact AR bijection over tokens) for transformer flow"
```

---

### Task 5: TransformerFlow + make_tarflow

**Files:**
- Create: `src/gensbi/normalizing_flows/transformer_flow/model.py`
- Test: `tests/normalizing_flows/transformer_flow/test_model.py`

**Interfaces:**
- Consumes: `VectorTokenizer` (T1), `VectorConditioner` (T2), `MetaBlock` (T4), `Mask` (base), `make_gaussian_prior` (`gensbi.core.prior`).
- Produces:
  - `TransformerFlow(blocks: list[MetaBlock], tokenizer: VectorTokenizer, dim, cond_dim, standardize=True)` (`nnx.Module`) with `log_prob(x: (B, dim), cond=None) -> (B,)`, `sample(key, cond=None, nsamples=None) -> (B, dim)`, `set_standardization(mean, std)`.
  - `make_tarflow(rngs, dim, cond_dim=0, channels=64, num_blocks=8, layers_per_block=2, head_dim=16, block_size=1, permutation="flip", standardize=True, zero_init=True) -> TransformerFlow`.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_model.py
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid
import pytest

from gensbi.normalizing_flows.transformer_flow.model import (
    TransformerFlow, make_tarflow,
)
from gensbi.core.prior import make_gaussian_prior


def _flow(dim=4, cond_dim=2, **kw):
    return make_tarflow(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, channels=16,
                        num_blocks=4, layers_per_block=2, head_dim=8, **kw)


def test_log_prob_shape_and_finite():
    flow = _flow()
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))


def test_zero_init_flow_is_standard_normal():
    dim, cond_dim = 4, 2
    flow = _flow(dim=dim, cond_dim=cond_dim, zero_init=True)
    base = make_gaussian_prior((dim,))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, dim))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, cond_dim))
    lp = flow.log_prob(x, cond)
    lp_base = jax.vmap(base.log_prob)(x)
    assert jnp.allclose(lp, lp_base, atol=1e-4)


def test_full_flow_logdet_matches_autodiff():
    dim, cond_dim = 4, 2
    flow = _flow(dim=dim, cond_dim=cond_dim, zero_init=False)
    cond = jnp.array([0.3, -0.4])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])
    base = make_gaussian_prior((dim,))

    def to_noise(x):
        # reproduce the data→noise map (no standardization set => identity)
        z = flow.tokenizer.tokenize(x[None])
        for blk in flow.blocks:
            z, _ = blk.inverse(z, cond[None])
        return z.reshape(-1)

    _, ad = jnp.linalg.slogdet(jax.jacobian(to_noise)(x))
    # analytic: log_prob = base.log_prob(z) + logdet  =>  logdet = lp - base
    z = to_noise(x)
    lp = flow.log_prob(x[None], cond[None])[0]
    analytic = lp - base.log_prob(z)
    assert jnp.allclose(ad, analytic, atol=1e-4)


def test_sample_shape_and_roundtrip_finite():
    flow = _flow(zero_init=False)
    cond = jnp.zeros((5, 2))
    s = flow.sample(jax.random.PRNGKey(3), cond=cond)
    assert s.shape == (5, 4)
    assert jnp.all(jnp.isfinite(flow.log_prob(s, cond)))


def test_density_integrates_to_one_2d():
    # NOTE: scalar-per-token with dim=1 is the identity (T=1 ⇒ shift-by-one
    # zeroes all params), so normalization is only meaningful for T>=2. Use 2-D.
    flow = make_tarflow(nnx.Rngs(0), dim=2, cond_dim=1, channels=16,
                        num_blocks=4, layers_per_block=2, head_dim=8,
                        zero_init=False)
    g = jnp.linspace(-8.0, 8.0, 161)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)        # (N, 2)
    cond = jnp.zeros((grid.shape[0], 1))
    dens = jnp.exp(flow.log_prob(grid, cond)).reshape(161, 161)
    integral = trapezoid(trapezoid(dens, g, axis=1), g)
    assert jnp.allclose(integral, 1.0, atol=2e-2)


def test_log_prob_depends_on_condition():
    flow = _flow(zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    lp_a = flow.log_prob(x, jnp.zeros((5, 2)))
    lp_b = flow.log_prob(x, jnp.ones((5, 2)))
    assert not jnp.allclose(lp_a, lp_b)


def test_unconditional_flow():
    flow = make_tarflow(nnx.Rngs(0), dim=3, cond_dim=0, channels=16,
                        num_blocks=2, layers_per_block=1, head_dim=8)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    assert flow.log_prob(x).shape == (6,)


def test_set_standardization():
    flow = _flow()
    mean = jnp.array([1.0, -2.0, 0.5, 0.0])
    std = jnp.array([2.0, 0.5, 3.0, 1.0])
    flow.set_standardization(mean, std)
    assert jnp.allclose(flow.mean[...], mean)
    assert jnp.allclose(flow.std[...], std)


def test_set_standardization_raises_when_disabled():
    flow = _flow(standardize=False)
    with pytest.raises(ValueError):
        flow.set_standardization(jnp.zeros(4), jnp.ones(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: FAIL (`ModuleNotFoundError ...transformer_flow.model`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/gensbi/normalizing_flows/transformer_flow/model.py
"""TransformerFlow: a transformer autoregressive normalizing flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

Batched-native ``(B, T, F)`` density model, sibling to the per-example
``Flow``/``Chain`` track. ``log_prob`` runs the parallel data→noise pass;
``sample`` runs the sequential noise→data pass. Base is a fixed ``N(0, I)`` over
the tokens (nvp mode).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.normalizing_flows.bijections.base import Mask
from gensbi.normalizing_flows.transformer_flow.blocks import MetaBlock
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner
from gensbi.normalizing_flows.transformer_flow.tokenizers import VectorTokenizer

_LOG2PI = jnp.log(2.0 * jnp.pi)


class TransformerFlow(nnx.Module):
    """Stack of MetaBlocks + tokenizer + standardization + N(0,I) base."""

    def __init__(self, blocks, tokenizer, dim, cond_dim, standardize=True):
        self.blocks = nnx.List(blocks)
        self.tokenizer = tokenizer
        self.dim = dim
        self.cond_dim = cond_dim
        self.T = tokenizer.T
        self.F = tokenizer.F
        self._standardize = standardize
        self.mean = Mask(jnp.zeros((dim,)))
        self.std = Mask(jnp.ones((dim,)))

    def _base_log_prob(self, z: Array) -> Array:
        # z: (B, T, F); standard normal over (T, F)
        return -0.5 * jnp.sum(z ** 2, axis=(1, 2)) - 0.5 * self.T * self.F * _LOG2PI

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        x = jnp.atleast_2d(x)
        u = (x - self.mean[...]) / self.std[...]              # standardize
        logdet = -jnp.sum(jnp.log(self.std[...]))            # scalar
        z = self.tokenizer.tokenize(u)                       # (B, T, F)
        total = jnp.broadcast_to(logdet, (x.shape[0],))
        for blk in self.blocks:
            z, ld = blk.inverse(z, cond)
            total = total + ld
        return self._base_log_prob(z) + total

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None):
        if cond is not None:
            nsamples = cond.shape[0]
        z = jax.random.normal(key, (nsamples, self.T, self.F))
        x = z
        for blk in reversed(self.blocks):
            x, _ = blk.forward(x, cond)
        x = self.tokenizer.detokenize(x)                     # (B, dim)
        return x * self.std[...] + self.mean[...]            # un-standardize

    def set_standardization(self, mean, std) -> None:
        if not self._standardize:
            raise ValueError(
                "TransformerFlow built with standardize=False")
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)


def make_tarflow(rngs, dim, cond_dim=0, channels=64, num_blocks=8,
                 layers_per_block=2, head_dim=16, block_size=1,
                 permutation="flip", standardize=True, zero_init=True):
    """Build a TransformerFlow stack (mirrors ``make_maf``)."""
    tokenizer = VectorTokenizer(dim, block_size)
    T, F = tokenizer.T, tokenizer.F
    blocks = []
    for i in range(num_blocks):
        if permutation == "flip":
            perm = jnp.arange(T) if i % 2 == 0 else jnp.arange(T)[::-1]
        elif permutation == "random":
            perm = jax.random.permutation(rngs.params(), T)
        else:
            raise ValueError(f"unknown permutation {permutation!r}")
        conditioner = VectorConditioner(cond_dim, channels, rngs=rngs)
        blocks.append(MetaBlock(
            F=F, channels=channels, T=T, perm=perm, inv_perm=jnp.argsort(perm),
            conditioner=conditioner, num_layers=layers_per_block,
            head_dim=head_dim, expansion=4, rngs=rngs, zero_init=zero_init))
    return TransformerFlow(blocks, tokenizer, dim, cond_dim,
                           standardize=standardize)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/model.py \
        tests/normalizing_flows/transformer_flow/test_model.py
git commit -m "feat(nf): TransformerFlow + make_tarflow (log_prob, sample, standardize)"
```

---

### Task 6: Package exports

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/__init__.py`
- Modify: `src/gensbi/normalizing_flows/__init__.py`
- Test: `tests/normalizing_flows/transformer_flow/test_exports.py`

**Interfaces:**
- Produces: `from gensbi.normalizing_flows import make_tarflow, TransformerFlow` works; and `from gensbi.normalizing_flows.transformer_flow import make_tarflow, TransformerFlow` works.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_exports.py
def test_top_level_exports():
    from gensbi.normalizing_flows import make_tarflow, TransformerFlow
    from gensbi.normalizing_flows.transformer_flow import (
        make_tarflow as mt2, TransformerFlow as TF2,
    )
    assert make_tarflow is mt2
    assert TransformerFlow is TF2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_exports.py -v`
Expected: FAIL (`ImportError: cannot import name 'make_tarflow'`).

- [ ] **Step 3: Write minimal implementation**

Set `src/gensbi/normalizing_flows/transformer_flow/__init__.py`:

```python
"""Transformer autoregressive normalizing flow (adapted from TarFlow).

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.
"""

from gensbi.normalizing_flows.transformer_flow.model import (
    TransformerFlow, make_tarflow,
)

__all__ = ["TransformerFlow", "make_tarflow"]
```

Then add to `src/gensbi/normalizing_flows/__init__.py` (read the file first; append the import and extend `__all__`):

```python
from gensbi.normalizing_flows.transformer_flow import TransformerFlow, make_tarflow
```

Add `"TransformerFlow"` and `"make_tarflow"` to that module's `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_exports.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/__init__.py \
        src/gensbi/normalizing_flows/transformer_flow/__init__.py \
        tests/normalizing_flows/transformer_flow/test_exports.py
git commit -m "feat(nf): export TransformerFlow/make_tarflow from normalizing_flows"
```

---

### Task 7: Pipeline + NLE integration (fast)

**Files:**
- Test: `tests/normalizing_flows/transformer_flow/test_pipeline_integration.py`

**Interfaces:**
- Consumes: `make_tarflow` (T5/T6); `ConditionalFlowPipeline` (`gensbi.recipes.flow_pipeline`); `NLEPosterior` (`gensbi.inference`); `make_gaussian_prior` (`gensbi.core.prior`).
- Produces: confirmation that `TransformerFlow` satisfies the pipeline/NLE duck-typed contract (no new source). If a test reveals a contract gap, fix it in `model.py` and note it.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_pipeline_integration.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain

from gensbi.normalizing_flows import make_tarflow
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior
from gensbi.core.prior import make_gaussian_prior

# NLE convention: obs = x (M-dim), cond = theta (D-dim)
M, D, N = 3, 2, 1024
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = jax.random.normal(_kth, (N, D))
_W = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])     # (M, D)
_x = _theta @ _W.T + 0.1 * jax.random.normal(_kx, (N, M))
DATA = jnp.concatenate([_x[..., None], _theta[..., None]], axis=1)  # (N, M+D, 1)


def _split(d):
    return d[:, :M], d[:, M:]            # (obs=x, cond=theta)


def _ds(arr, bs=128):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(bs).map(_split))


def _pipe(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), dim=M, cond_dim=D, channels=16,
                        num_blocks=4, layers_per_block=2, head_dim=8,
                        standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    return ConditionalFlowPipeline(flow, _ds(DATA[:800]), _ds(DATA[800:]),
                                   M, D, ch_obs=1, ch_cond=1,
                                   training_config=cfg)


def test_loss_scalar_and_finite(tmp_path):
    pipe = _pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :M])
    cond = jnp.asarray(DATA[:32, M:])
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == () and jnp.isfinite(loss)


def test_grads_flow_to_params(tmp_path):
    pipe = _pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs, cond = jnp.asarray(DATA[:32, :M]), jnp.asarray(DATA[:32, M:])
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


def test_fit_standardization_sets_both_models(tmp_path):
    pipe = _pipe(tmp_path)
    pipe.fit_standardization(DATA[:800, :M])     # standardize x
    exp_mean = jnp.mean(DATA[:800, :M, 0], axis=0)
    for flow in (pipe.model, pipe.ema_model):
        assert jnp.allclose(flow.mean[...], exp_mean, atol=1e-4)
    assert pipe._standardized is True


def test_train_smoke_and_log_prob(tmp_path):
    pipe = _pipe(tmp_path)
    pipe.fit_standardization(DATA[:800, :M])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    x_1 = jnp.zeros((5, M, 1))
    x_o = jnp.zeros((1, D, 1))
    lp = pipe.log_prob(x_1, x_o, use_ema=False)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))


def test_nle_potential_value_and_grad(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), dim=M, cond_dim=D, channels=16,
                        num_blocks=3, layers_per_block=1, head_dim=8,
                        zero_init=False)
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(flow, prior)
    U = post.potential(jnp.array([0.5, -0.5, 0.2]))
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_pipeline_integration.py -v`
Expected: FAIL initially only if a contract gap exists; otherwise this is the verification gate. If any test errors on a shape/contract mismatch, fix `TransformerFlow` in `model.py` (e.g. ensure `log_prob` accepts `(B, dim)` and `set_standardization`/`mean`/`std` names match) and re-run.

- [ ] **Step 3: Fix any contract gaps (if needed)**

If `test_*` reveals a mismatch, the minimal fix lives in `model.py`. Most likely none is needed — `log_prob(x, cond)`, `sample(key, cond, nsamples)`, `set_standardization(mean, std)`, and the `mean`/`std` buffers already match the `ConditionalFlowPipeline`/`NLEPosterior` contract used by the MAF flow.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_pipeline_integration.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/normalizing_flows/transformer_flow/test_pipeline_integration.py \
        src/gensbi/normalizing_flows/transformer_flow/model.py
git commit -m "test(nf): TransformerFlow integrates with ConditionalFlowPipeline + NLEPosterior"
```

---

### Task 8: End-to-end NLE recovery (slow)

**Files:**
- Test: `tests/normalizing_flows/transformer_flow/test_nle_e2e.py`

**Interfaces:**
- Consumes: `make_tarflow`, `ConditionalFlowPipeline`, `NLEPosterior`, `make_gaussian_prior`.
- Produces: a `@pytest.mark.slow` test proving the NLE TransformerFlow recovers an analytic linear-Gaussian posterior (mirrors `tests/normalizing_flows/test_nle_e2e.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_nle_e2e.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows import make_tarflow
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior

D, M, SIGMA = 2, 3, 0.5
G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])   # (M, D)


def _simulate(key, n):
    kth, ke = jax.random.split(key)
    theta = jax.random.normal(kth, (n, D))
    x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
    return theta, x


def _analytic_posterior(x_o):
    prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
    cov = jnp.linalg.inv(prec)
    mean = cov @ (G.T @ x_o) / SIGMA ** 2
    return mean, cov


@pytest.mark.slow
def test_tarflow_nle_recovers_linear_gaussian(tmp_path):
    theta, x = _simulate(jax.random.PRNGKey(0), 20_000)
    data = jnp.concatenate([x[..., None], theta[..., None]], axis=1)  # x FIRST

    def split(d):
        return d[:, :M], d[:, M:]            # (obs=x, cond=theta)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split))

    flow = make_tarflow(nnx.Rngs(0), dim=M, cond_dim=D, channels=64,
                        num_blocks=6, layers_per_block=2, head_dim=16,
                        standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=4000, val_every=200, max_lr=3e-4,
                    checkpoint_dir=str(tmp_path), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(data[:18_000]),
                                   make_ds(data[18_000:]), M, D,
                                   ch_obs=1, ch_cond=1, training_config=cfg)
    pipe.fit_standardization(data[:18_000, :M])     # standardize x
    pipe.train(nnx.Rngs(0), nsteps=4000, save_model=False)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(pipe.ema_model, prior, num_warmup=500, num_samples=4000)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]   # (n, D)

    assert jnp.allclose(jnp.mean(s, axis=0), mean_a, atol=0.2), (jnp.mean(s, 0), mean_a)
    assert jnp.allclose(jnp.cov(s.T), cov_a, atol=0.2)
```

- [ ] **Step 2: Run test to verify it fails (or is collected)**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_nle_e2e.py -v -m slow`
Expected: FAIL if recovery is off (tune `num_blocks`/`nsteps`/`max_lr` within reason); the deliverable is a passing recovery.

- [ ] **Step 3: Tune if needed**

If the posterior moments miss the analytic targets, increase `nsteps` (e.g. 6000) or `num_blocks` (e.g. 8) — do NOT loosen `atol` below the mirrored MAF test's 0.15–0.2 band. The flow math is already verified in Tasks 4–5; a miss here is capacity/optimization, not correctness.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_nle_e2e.py -v -m slow`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/normalizing_flows/transformer_flow/test_nle_e2e.py
git commit -m "test(nf): e2e NLE TransformerFlow recovers linear-Gaussian posterior (slow)"
```

---

## Final verification

Run the full transformer-flow suite (fast) and the whole NF package to confirm no regressions:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow -v
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows -q
```

Expected: all fast tests pass; the one slow e2e test passes under `-m slow`.
