# TarFlow (TransformerFlow) v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v1 `TransformerFlow` with an invertible `ImageTokenizer` (field-level NLE on image/field `x`) and STARFlow-style **prefix-concatenation** conditioning (vector *or* image conditions), plus the pipeline/inference plumbing so both new setups train and sample end-to-end.

**Architecture:** The modeled-variable tokenizer and the conditioner are orthogonal seams feeding a stack of `MetaBlock`s. v2 adds: (a) `ImageTokenizer` over `patchify_2d`; (b) a unified conditioner contract `embed(cond) -> (bias, prefix)` with a new `PrefixConditioner` family (vector/image encoders → prefix tokens); (c) a **uniform SOS input-shift** in `MetaBlock` (replacing v1's post-`proj_out` zero-shift) so token 0 is conditioned; (d) a prefix-causal attention mask. The Jacobian-critical invariants (`logdet = −Σa`, triangular structure) are preserved and re-verified.

**Tech Stack:** JAX, Flax NNX, `jax.nn.dot_product_attention`, `gensbi.recipes.utils.patchify_2d/depatchify_2d`, NumPyro (NUTS via `NLEPosterior`), grain (datasets in tests), pytest.

Design spec: `docs/superpowers/specs/2026-06-24-tarflow-nle-v2-design.md`. References: `reference/ml-tarflow/transformer_flow.py` (v1 base), `reference/ml-starflow/transformer_flow.py` (prefix-concat + SOS).

## Global Constraints

- **Precision:** float32 throughout; accumulate log-dets in float32. (Matches v1.)
- **Run tests with:** `JAX_PLATFORMS=cpu .venv/bin/python -m pytest` (GPUs usually busy; `cpu` is what the NF tests pin).
- **Module location:** `src/gensbi/normalizing_flows/transformer_flow/` (production NF track).
- **NNX conventions:** learnable tensors are `nnx.Param`; fixed buffers (permutations, standardization mean/std) are `gensbi.normalizing_flows.bijections.base.Mask` (excluded from EMA, still checkpointed).
- **Direction convention:** `inverse` = data→noise (density, parallel pass); `forward` = noise→data (sampling, sequential scan). Unchanged from v1.
- **Exactness invariants (must not regress):** (1) token *i*'s affine params depend only on modeled tokens `< i` **and** the condition (causal mask + SOS input-shift) ⇒ `log|det| = −Σ a` over modeled tokens; (2) the condition (bias or prefix) is a function of the condition **only** — prefix tokens never attend to modeled tokens (`cond→x` blocked in the mask); (3) normalize over the **channel** axis only, never across tokens (incl. the prefix↔modeled boundary); (4) the modeled-variable tokenizer is a volume-preserving reshape (log-det 0); (5) `zero_init` ⇒ each block is identity ⇒ the flow starts at the standardized-Gaussian base (SOS does not change this — zero affine params ⇒ identity regardless of the hidden state).
- **Conditioner contract (v2):** every conditioner exposes `embed(cond) -> tuple[Array | None, Array | None]` returning `(bias, prefix)`. At most one is non-`None`. `bias` is `(B, channels)` (broadcast-added per token); `prefix` is `(B, M, channels)` (prepended as prefix tokens). Both `None` ⇒ unconditional. The v1 `inject` method is removed.
- **Attribution:** v1 files keep `Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.`. Files implementing prefix-concat/SOS additionally get: `Prefix-concatenation conditioning and SOS shift adapted from apple/ml-starflow (STARFlow); see transformer_flow/LICENSE.starflow.`. Task 3 creates `LICENSE.starflow` (STARFlow's permissive code license; its restrictive `LICENSE_MODEL` governs released weights, not this clean-room architecture reimplementation).

---

### Task 1: ImageTokenizer (invertible patchify seam) + tokenizer `example_shape`

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py`
- Test: `tests/normalizing_flows/transformer_flow/test_tokenizers.py` (append)

**Interfaces:**
- Consumes: `gensbi.recipes.utils.patchify_2d`, `depatchify_2d`.
- Produces:
  - `VectorTokenizer` gains attribute `.example_shape = (dim,)`.
  - `ImageTokenizer(height, width, channels, patch_size)` with attributes `.height`, `.width`, `.channels`, `.patch_size`, `.grid = (height//patch_size, width//patch_size)`, `.T = grid[0]*grid[1]`, `.F = channels*patch_size**2`, `.example_shape = (height, width, channels)`; methods `tokenize(x: (B,H,W,C)) -> (B,T,F)` and `detokenize(tokens: (B,T,F)) -> (B,H,W,C)`. Plain class (no params).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/normalizing_flows/transformer_flow/test_tokenizers.py
import jax
from gensbi.normalizing_flows.transformer_flow.tokenizers import ImageTokenizer
from gensbi.recipes.utils import patchify_2d


def test_vector_tokenizer_example_shape():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert tok.example_shape == (6,)


def test_image_tokenizer_shapes():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    assert (tok.T, tok.F) == (16, 8)          # T=(8/2)^2=16, F=2*2*2=8
    assert tok.example_shape == (8, 8, 2)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 8, 2))
    assert tok.tokenize(x).shape == (3, 16, 8)


def test_image_tokenizer_matches_patchify_2d():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 8, 2))
    assert jnp.allclose(tok.tokenize(x), patchify_2d(x, size=2))


def test_image_tokenizer_roundtrip():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 8, 8, 2))
    assert jnp.allclose(tok.detokenize(tok.tokenize(x)), x, atol=1e-6)


def test_image_tokenizer_non_divisible_raises():
    with pytest.raises(ValueError):
        ImageTokenizer(height=7, width=8, channels=1, patch_size=2)
```

(Ensure `import jax.numpy as jnp` and `import pytest` are present at the top of the file; they are from v1.)

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_tokenizers.py -v`
Expected: FAIL (`ImportError: cannot import name 'ImageTokenizer'`).

- [ ] **Step 3: Write minimal implementation**

Append to `src/gensbi/normalizing_flows/transformer_flow/tokenizers.py`, and add `example_shape` to `VectorTokenizer.__init__`:

```python
# add at top of file
from gensbi.recipes.utils import patchify_2d, depatchify_2d

# inside VectorTokenizer.__init__, after self.T = dim // block_size:
        self.example_shape = (dim,)
```

```python
# append ImageTokenizer
class ImageTokenizer:
    """Patchify an image ``(B, H, W, C)`` into ``T = (H/p)(W/p)`` tokens of
    ``F = C*p*p`` features via :func:`patchify_2d`. Pure reshape: invertible,
    log-det 0. Raster causal order (fixed by ``patchify_2d``)."""

    def __init__(self, height: int, width: int, channels: int, patch_size: int):
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must divide height ({height}) and "
                f"width ({width})")
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.grid = (height // patch_size, width // patch_size)
        self.T = self.grid[0] * self.grid[1]
        self.F = channels * patch_size * patch_size
        self.example_shape = (height, width, channels)

    def tokenize(self, x: Array) -> Array:
        return patchify_2d(x, size=self.patch_size)

    def detokenize(self, tokens: Array) -> Array:
        return depatchify_2d(tokens, size=self.patch_size, grid=self.grid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_tokenizers.py -v`
Expected: PASS (all tokenizer tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/tokenizers.py \
        tests/normalizing_flows/transformer_flow/test_tokenizers.py
git commit -m "feat(nf): ImageTokenizer (invertible patchify seam) + tokenizer example_shape"
```

---

### Task 2: AttentionBlock optional mask argument

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (`AttentionBlock.__call__`)
- Test: `tests/normalizing_flows/transformer_flow/test_blocks_attention.py` (append)

**Interfaces:**
- Produces: `AttentionBlock.__call__(x: (B,T,C), mask: (S,S) bool | None = None) -> (B,T,C)`. `mask is None` ⇒ `is_causal=True` (v1 behavior). A `(S,S)` boolean mask (`True` = attend) is reshaped to `(1,1,S,S)` and passed as `mask=` (with `is_causal` off).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/normalizing_flows/transformer_flow/test_blocks_attention.py
def test_explicit_tril_mask_matches_is_causal():
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    T = 5
    x = jax.random.normal(jax.random.PRNGKey(3), (2, T, 8))
    tril = jnp.tril(jnp.ones((T, T), dtype=bool))
    assert jnp.allclose(blk(x), blk(x, tril), atol=1e-6)


def test_prefix_mask_blocks_prefix_from_seeing_modeled():
    """With a prefix-LM mask, prefix-row outputs must not depend on modeled inputs."""
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    M, T = 2, 3
    S = M + T
    idx = jnp.arange(S)
    is_modeled_q = idx[:, None] >= M
    is_prefix_k = idx[None, :] < M
    causal = idx[None, :] <= idx[:, None]
    mask = jnp.where(is_modeled_q, is_prefix_k | causal, is_prefix_k)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (S, 8))

    def f(x):
        return blk(x[None], mask)[0]            # (S, C)

    J = jax.jacrev(f)(x0)                        # (S, C, S, C)
    # prefix rows (i < M) must be invariant to modeled inputs (j >= M)
    for i in range(M):
        for j in range(M, S):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_attention.py -v`
Expected: FAIL (`__call__` takes no `mask` argument / `TypeError`).

- [ ] **Step 3: Write minimal implementation**

Replace `AttentionBlock.__call__` in `blocks.py`:

```python
    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, hd)
        if mask is None:
            attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = jax.nn.dot_product_attention(q, k, v, mask=mask[None, None])
        x = x + self.proj(attn.reshape(B, T, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_attention.py -v`
Expected: PASS (4 passed — 2 v1 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py \
        tests/normalizing_flows/transformer_flow/test_blocks_attention.py
git commit -m "feat(nf): AttentionBlock optional attention mask (prefix-causal support)"
```

---

### Task 3: Conditioner contract `(bias, prefix)` + uniform SOS input-shift in MetaBlock

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/conditioners.py` (`VectorConditioner`)
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (`MetaBlock`)
- Create: `src/gensbi/normalizing_flows/transformer_flow/LICENSE.starflow`
- Test: `tests/normalizing_flows/transformer_flow/test_conditioners.py` (rewrite cases for the new contract)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `VectorConditioner.embed(cond) -> (bias: (B,channels) | None, None)`; `cond_dim == 0` ⇒ `(None, None)`. The `inject` method is **removed**.
  - `MetaBlock` gains a learned `sos_embed` `nnx.Param` of shape `(1, 1, F)`; `_params(x_perm, cond)` now applies the SOS input-shift and consumes `(bias, prefix)` from `embed` (the **prefix branch is added in Task 5**; here `_params` handles `bias`/unconditional only and the post-`proj_out` zero-shift is removed).

- [ ] **Step 1: Write the failing test**

```python
# rewrite tests/normalizing_flows/transformer_flow/test_conditioners.py
import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner


def test_embed_returns_bias_prefix_tuple():
    cond_dim, channels, B = 3, 8, 4
    c = VectorConditioner(cond_dim, channels, rngs=nnx.Rngs(0))
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim))
    bias, prefix = c.embed(cond)
    assert bias.shape == (B, channels)
    assert prefix is None


def test_unconditional_returns_none_none():
    c = VectorConditioner(0, 8, rngs=nnx.Rngs(0))
    assert c.embed(jnp.zeros((4, 0))) == (None, None)


def test_missing_cond_raises():
    c = VectorConditioner(3, 8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        c.embed(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py -v`
Expected: FAIL (`embed` returns a single array, not a tuple).

- [ ] **Step 3: Write minimal implementation**

In `conditioners.py`, replace `VectorConditioner.embed` and delete `inject`:

```python
    def embed(self, cond: Array | None):
        """Return ``(bias, prefix)``; VectorConditioner only sets ``bias``."""
        if self.cond_dim == 0:
            return (None, None)
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        bias = self.l2(jax.nn.silu(self.l1(cond)))
        return (bias, None)
```

In `blocks.py`, add the STARFlow attribution line to the module docstring and update `MetaBlock`. Add `sos_embed` in `__init__` (after `self.proj_in = ...`):

```python
        self.sos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, F)) * 1e-2)
```

Replace `MetaBlock._params` with (SOS input-shift; **no** post-`proj_out` shift; bias/unconditional only — prefix added in Task 5):

```python
    def _params(self, x_perm: Array, cond: Array | None):
        """(a, b) for the permuted tokens. SOS input-shift makes token i's
        params depend only on tokens < i (and the condition)."""
        bias, prefix = self.conditioner.embed(cond)
        B = x_perm.shape[0]
        sos = jnp.broadcast_to(self.sos_embed[...], (B, 1, self.F))
        x_in = jnp.concatenate([sos, x_perm[:, :-1]], axis=1)   # (B, T, F)
        h = self.proj_in(x_in) + self.pos_embed[...][None]      # (B, T, C)
        if bias is not None:
            h = h + bias[:, None, :]
        for blk in self.attn_blocks:
            h = blk(h)                                          # is_causal
        out = self.proj_out(h)                                  # (B, T, 2F)
        a, b = jnp.split(out, 2, axis=-1)                       # each (B, T, F)
        return a, b
```

Add the STARFlow attribution to the top of `blocks.py` docstring:

```python
"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.
Prefix-concatenation conditioning and SOS shift adapted from apple/ml-starflow
(STARFlow); see transformer_flow/LICENSE.starflow.
"""
```

Create `src/gensbi/normalizing_flows/transformer_flow/LICENSE.starflow`:

```text
This module's prefix-concatenation conditioning and SOS input-shift are a
clean-room JAX reimplementation of ideas in the STARFlow PyTorch code released
by Apple:

    apple/ml-starflow  —  https://github.com/apple/ml-starflow

Apple's source `LICENSE` (a permissive, BSD-style grant: use, reproduce, modify,
redistribute provided this notice is retained; Apple's marks may not be used to
endorse derived works) governs adaptation of the source. The repository's
separate `LICENSE_MODEL` is research-only and applies to Apple's released model
weights, NOT to this independent architecture reimplementation, which uses no
Apple weights. Apple's copyright notice is retained here in acknowledgement.

    Copyright (C) 2025 Apple Inc. All Rights Reserved.
```

- [ ] **Step 4: Run test to verify it passes (incl. v1 regression)**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py tests/normalizing_flows/transformer_flow/test_blocks_meta.py tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: PASS. The v1 `MetaBlock`/model tests are structural (zero-init identity, triangular, logdet-vs-autodiff, roundtrip, normalization) and still hold under SOS — only token 0's *learned* behavior changes. If `test_blocks_meta.py` or `test_model.py` reference the removed `inject` or a non-tuple `embed`, that is a real break: fix the test to the new contract (it should not — only `MetaBlock` calls them internally).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/conditioners.py \
        src/gensbi/normalizing_flows/transformer_flow/blocks.py \
        src/gensbi/normalizing_flows/transformer_flow/LICENSE.starflow \
        tests/normalizing_flows/transformer_flow/test_conditioners.py
git commit -m "refactor(nf): conditioner (bias,prefix) contract + uniform SOS input-shift"
```

---

### Task 4: PrefixConditioner family (vector + image encoders → prefix tokens)

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/conditioners.py` (append two classes)
- Test: `tests/normalizing_flows/transformer_flow/test_conditioners.py` (append)

**Interfaces:**
- Consumes: `gensbi.recipes.utils.patchify_2d`.
- Produces (both `nnx.Module`, both `embed(cond) -> (None, prefix: (B, M, channels))`, both expose `.M`):
  - `VectorPrefixConditioner(cond_dim: int, channels: int, num_tokens: int, rngs)` — `Linear(cond_dim -> channels*num_tokens)` reshaped to `(B, num_tokens, channels)` + a learned `(num_tokens, channels)` pos-embed.
  - `ImagePrefixConditioner(cond_channels: int, patch_size: int, channels: int, num_tokens: int, rngs)` — `patchify_2d(cond, size=patch_size)` → `Linear(cond_channels*patch_size**2 -> channels)` + learned `(num_tokens, channels)` pos-embed. `num_tokens` must equal `(H/p)(W/p)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/normalizing_flows/transformer_flow/test_conditioners.py
from gensbi.normalizing_flows.transformer_flow.conditioners import (
    VectorPrefixConditioner, ImagePrefixConditioner,
)


def test_vector_prefix_shapes():
    c = VectorPrefixConditioner(cond_dim=3, channels=8, num_tokens=2, rngs=nnx.Rngs(0))
    assert c.M == 2
    cond = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    bias, prefix = c.embed(cond)
    assert bias is None
    assert prefix.shape == (4, 2, 8)


def test_image_prefix_shapes():
    # cond image 8x8x2, patch 2 -> M = 16 tokens
    c = ImagePrefixConditioner(cond_channels=2, patch_size=2, channels=8,
                               num_tokens=16, rngs=nnx.Rngs(0))
    assert c.M == 16
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 8, 8, 2))
    bias, prefix = c.embed(cond)
    assert bias is None
    assert prefix.shape == (4, 16, 8)


def test_prefix_depends_on_condition():
    c = VectorPrefixConditioner(cond_dim=3, channels=8, num_tokens=1, rngs=nnx.Rngs(0))
    _, p1 = c.embed(jnp.zeros((2, 3)))
    _, p2 = c.embed(jnp.ones((2, 3)))
    assert not jnp.allclose(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Append to `conditioners.py` (and add `from gensbi.recipes.utils import patchify_2d` at top):

```python
class VectorPrefixConditioner(nnx.Module):
    """Vector condition → ``num_tokens`` prefix tokens ``(B, M, channels)``."""

    def __init__(self, cond_dim: int, channels: int, num_tokens: int, rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        self.channels = channels
        self.M = num_tokens
        self.proj = nnx.Linear(cond_dim, channels * num_tokens, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, channels)) * 1e-2)

    def embed(self, cond: Array | None):
        if cond is None:
            raise ValueError("cond is required for VectorPrefixConditioner")
        B = cond.shape[0]
        h = self.proj(cond).reshape(B, self.M, self.channels)
        return (None, h + self.pos[...][None])


class ImagePrefixConditioner(nnx.Module):
    """Image condition ``(B, H, W, C)`` → ``M = (H/p)(W/p)`` prefix tokens."""

    def __init__(self, cond_channels: int, patch_size: int, channels: int,
                 num_tokens: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.channels = channels
        self.M = num_tokens
        in_f = cond_channels * patch_size * patch_size
        self.proj = nnx.Linear(in_f, channels, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, channels)) * 1e-2)

    def embed(self, cond: Array | None):
        if cond is None:
            raise ValueError("cond is required for ImagePrefixConditioner")
        patches = patchify_2d(cond, size=self.patch_size)      # (B, M, in_f)
        return (None, self.proj(patches) + self.pos[...][None])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_conditioners.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/conditioners.py \
        tests/normalizing_flows/transformer_flow/test_conditioners.py
git commit -m "feat(nf): PrefixConditioner family (vector + image encoders to prefix tokens)"
```

---

### Task 5: MetaBlock prefix-concat path (prefix-LM mask + strip)

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/blocks.py` (`MetaBlock`)
- Test: `tests/normalizing_flows/transformer_flow/test_blocks_meta.py` (append)

**Interfaces:**
- Consumes: `AttentionBlock.__call__(x, mask)` (Task 2); `VectorPrefixConditioner` (Task 4).
- Produces: `MetaBlock._params` now handles the `prefix` branch — when `embed` returns a `prefix`, it is prepended, attention runs under a prefix-LM mask (`tril` among modeled, prefix bidirectional, `cond→x` blocked), and stripped before `proj_out`. Adds method `MetaBlock._prefix_mask(M, T) -> (M+T, M+T) bool`. `inverse`/`forward` signatures unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/normalizing_flows/transformer_flow/test_blocks_meta.py
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorPrefixConditioner


def _make_prefix(T=4, F=1, channels=8, cond_dim=2, num_tokens=2, zero_init=False,
                 rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)
    cond = VectorPrefixConditioner(cond_dim, channels, num_tokens, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     inv_perm=jnp.argsort(perm), conditioner=cond, num_layers=2,
                     head_dim=4, expansion=2, rngs=rngs, zero_init=zero_init)


def test_prefix_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i, with a prefix condition."""
    blk = _make_prefix(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    J = jax.jacrev(f)(x0[:, 0])
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_prefix_logdet_matches_autodiff():
    blk = _make_prefix(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_prefix_roundtrip():
    blk = _make_prefix(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)


def test_prefix_zero_init_identity():
    blk = _make_prefix(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(7), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(8), (3, 2))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_meta.py -v`
Expected: FAIL (`embed` returns a `prefix` the current `_params` ignores ⇒ a shape error when concatenating, or a non-triangular/incorrect result).

- [ ] **Step 3: Write minimal implementation**

Add `_prefix_mask` to `MetaBlock` and extend `_params` with the prefix branch:

```python
    def _prefix_mask(self, M: int, T: int) -> Array:
        """Prefix-LM mask over [prefix(M); modeled(T)]: modeled is causal and
        sees all prefix; prefix is bidirectional among itself and never sees
        modeled (cond→x blocked)."""
        S = M + T
        idx = jnp.arange(S)
        is_modeled_q = idx[:, None] >= M
        is_prefix_k = idx[None, :] < M
        causal = idx[None, :] <= idx[:, None]
        return jnp.where(is_modeled_q, is_prefix_k | causal, is_prefix_k)
```

```python
    def _params(self, x_perm: Array, cond: Array | None):
        """(a, b) for the permuted tokens. SOS input-shift makes token i's
        params depend only on tokens < i (and the condition)."""
        bias, prefix = self.conditioner.embed(cond)
        B = x_perm.shape[0]
        sos = jnp.broadcast_to(self.sos_embed[...], (B, 1, self.F))
        x_in = jnp.concatenate([sos, x_perm[:, :-1]], axis=1)   # (B, T, F)
        h = self.proj_in(x_in) + self.pos_embed[...][None]      # (B, T, C)
        if bias is not None:
            h = h + bias[:, None, :]
        if prefix is not None:
            M = prefix.shape[1]
            h = jnp.concatenate([prefix, h], axis=1)            # (B, M+T, C)
            mask = self._prefix_mask(M, self.T)
            for blk in self.attn_blocks:
                h = blk(h, mask)
            h = h[:, M:]                                        # (B, T, C) strip
        else:
            for blk in self.attn_blocks:
                h = blk(h)
        out = self.proj_out(h)                                  # (B, T, 2F)
        a, b = jnp.split(out, 2, axis=-1)
        return a, b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_blocks_meta.py -v`
Expected: PASS (v1 meta tests + 4 new prefix tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/blocks.py \
        tests/normalizing_flows/transformer_flow/test_blocks_meta.py
git commit -m "feat(nf): MetaBlock prefix-concat conditioning (prefix-LM mask + strip)"
```

---

### Task 6: TransformerFlow + make_tarflow wiring (image modeled var & structured conditioning)

**Files:**
- Modify: `src/gensbi/normalizing_flows/transformer_flow/model.py`
- Test: `tests/normalizing_flows/transformer_flow/test_model.py` (append)

**Interfaces:**
- Consumes: `ImageTokenizer` (T1), `VectorPrefixConditioner`/`ImagePrefixConditioner` (T4), `MetaBlock` (T5).
- Produces:
  - `TransformerFlow.__init__` derives standardization buffer shape from `tokenizer.example_shape` (was hard-coded `(dim,)`); adds `self.example_shape`. `log_prob`/`sample` use `_ensure_batched` and broadcast `mean`/`std` over `example_shape`. `set_standardization(mean, std)` unchanged (shape now matches `example_shape`).
  - `make_tarflow(rngs, dim=None, cond_dim=0, *, modeled="vector", img_size=None, patch_size=None, img_channels=1, cond="add", cond_img_size=None, cond_patch_size=None, cond_channels=1, prefix_tokens=1, channels=64, num_blocks=8, layers_per_block=2, head_dim=16, block_size=1, permutation="flip", standardize=True, zero_init=True)`. `modeled ∈ {"vector","image"}`; `cond ∈ {"add","vector_prefix","image_prefix"}`. Vector path defaults reproduce v1 exactly.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/normalizing_flows/transformer_flow/test_model.py
def test_image_modeled_log_prob_and_sample():
    flow = make_tarflow(nnx.Rngs(0), cond_dim=2, modeled="image", img_size=8,
                        patch_size=2, img_channels=1, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (5, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))
    s = flow.sample(jax.random.PRNGKey(3), cond=cond)
    assert s.shape == (5, 8, 8, 1)


def test_image_modeled_zero_init_is_base():
    flow = make_tarflow(nnx.Rngs(0), cond_dim=2, modeled="image", img_size=8,
                        patch_size=2, img_channels=1, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8, zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8, 8, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
    lp = flow.log_prob(x, cond)
    # zero-init ⇒ identity flow ⇒ standard normal over the 8*8*1 elements
    expected = -0.5 * jnp.sum(x ** 2, axis=(1, 2, 3)) - 0.5 * 64 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(lp, expected, atol=1e-4)


def test_image_condition_npe_depends_on_condition():
    # NPE: modeled theta vector (dim=2), condition = 8x8x1 image via prefix
    flow = make_tarflow(nnx.Rngs(0), dim=2, modeled="vector", cond="image_prefix",
                        cond_img_size=8, cond_patch_size=2, cond_channels=1,
                        channels=16, num_blocks=4, layers_per_block=2, head_dim=8,
                        zero_init=False)
    theta = jax.random.normal(jax.random.PRNGKey(1), (5, 2))
    img_a = jnp.zeros((5, 8, 8, 1))
    img_b = jnp.ones((5, 8, 8, 1))
    assert not jnp.allclose(flow.log_prob(theta, img_a), flow.log_prob(theta, img_b))
    s = flow.sample(jax.random.PRNGKey(4), cond=img_a)
    assert s.shape == (5, 2)


def test_vector_path_unchanged():
    # the v1 default vector path still builds and runs
    flow = make_tarflow(nnx.Rngs(0), dim=4, cond_dim=2, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8)
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    cond = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    assert flow.log_prob(x, cond).shape == (8,)


def test_image_set_standardization_shape():
    flow = make_tarflow(nnx.Rngs(0), cond_dim=2, modeled="image", img_size=8,
                        patch_size=2, img_channels=1, channels=16, num_blocks=2,
                        layers_per_block=1, head_dim=8)
    mean = jnp.zeros((8, 8, 1))
    std = jnp.ones((8, 8, 1)) * 2.0
    flow.set_standardization(mean, std)
    assert flow.mean[...].shape == (8, 8, 1)
    assert jnp.allclose(flow.std[...], std)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: FAIL (`make_tarflow` has no `modeled`/`cond` kwargs; standardization buffers are `(dim,)`).

- [ ] **Step 3: Write minimal implementation**

In `model.py`: update imports, `TransformerFlow.__init__`, `_ensure_batched`, `log_prob`, `sample`, and `make_tarflow`.

```python
# imports (add)
from gensbi.normalizing_flows.transformer_flow.tokenizers import (
    VectorTokenizer, ImageTokenizer,
)
from gensbi.normalizing_flows.transformer_flow.conditioners import (
    VectorConditioner, VectorPrefixConditioner, ImagePrefixConditioner,
)
```

```python
# TransformerFlow.__init__ — replace the mean/std lines
    def __init__(self, blocks, tokenizer, dim, cond_dim, standardize=True):
        self.blocks = nnx.List(blocks)
        self.tokenizer = tokenizer
        self.dim = dim
        self.cond_dim = cond_dim
        self.T = tokenizer.T
        self.F = tokenizer.F
        self.example_shape = tokenizer.example_shape
        self._standardize = standardize
        self.mean = Mask(jnp.zeros(self.example_shape))
        self.std = Mask(jnp.ones(self.example_shape))

    def _ensure_batched(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == len(self.example_shape):
            x = x[None]
        return x
```

```python
# TransformerFlow.log_prob — replace the first two lines
    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        x = self._ensure_batched(x)
        u = (x - self.mean[...]) / self.std[...]              # standardize
        logdet = -jnp.sum(jnp.log(self.std[...]))            # over all elements
        z = self.tokenizer.tokenize(u)                       # (B, T, F)
        total = jnp.broadcast_to(logdet, (x.shape[0],))
        for blk in self.blocks:
            z, ld = blk.inverse(z, cond)
            total = total + ld
        return self._base_log_prob(z) + total
```

(`sample` and `set_standardization` need no change: `detokenize` returns `(B, *example_shape)` and `* std + mean` broadcasts; `set_standardization` assigns arrays whose shape already matches the buffers.)

Replace `make_tarflow`:

```python
def make_tarflow(rngs, dim=None, cond_dim=0, *, modeled="vector",
                 img_size=None, patch_size=None, img_channels=1,
                 cond="add", cond_img_size=None, cond_patch_size=None,
                 cond_channels=1, prefix_tokens=1,
                 channels=64, num_blocks=8, layers_per_block=2, head_dim=16,
                 block_size=1, permutation="flip", standardize=True,
                 zero_init=True):
    """Build a TransformerFlow stack. ``modeled`` selects the tokenizer
    (vector/image); ``cond`` selects the conditioner (additive bias /
    vector-prefix / image-prefix). Vector defaults reproduce v1."""
    if modeled == "vector":
        tokenizer = VectorTokenizer(dim, block_size)
    elif modeled == "image":
        tokenizer = ImageTokenizer(img_size, img_size, img_channels, patch_size)
    else:
        raise ValueError(f"unknown modeled {modeled!r}")
    T, F = tokenizer.T, tokenizer.F

    def make_cond():
        if cond == "add":
            return VectorConditioner(cond_dim, channels, rngs=rngs)
        if cond == "vector_prefix":
            return VectorPrefixConditioner(cond_dim, channels, prefix_tokens,
                                           rngs=rngs)
        if cond == "image_prefix":
            m = (cond_img_size // cond_patch_size) ** 2
            return ImagePrefixConditioner(cond_channels, cond_patch_size,
                                          channels, m, rngs=rngs)
        raise ValueError(f"unknown cond {cond!r}")

    blocks = []
    for i in range(num_blocks):
        if permutation == "flip":
            perm = jnp.arange(T) if i % 2 == 0 else jnp.arange(T)[::-1]
        elif permutation == "random":
            perm = jax.random.permutation(rngs.params(), T)
        else:
            raise ValueError(f"unknown permutation {permutation!r}")
        blocks.append(MetaBlock(
            F=F, channels=channels, T=T, perm=perm, inv_perm=jnp.argsort(perm),
            conditioner=make_cond(), num_layers=layers_per_block,
            head_dim=head_dim, expansion=4, rngs=rngs, zero_init=zero_init))
    return TransformerFlow(blocks, tokenizer, dim, cond_dim,
                           standardize=standardize)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_model.py -v`
Expected: PASS (v1 model tests + 5 new). Then run the whole transformer-flow suite to confirm no regression:
`JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow -q`

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/transformer_flow/model.py \
        tests/normalizing_flows/transformer_flow/test_model.py
git commit -m "feat(nf): TransformerFlow image modeled var + structured conditioning wiring"
```

---

### Task 7: Pipeline structured-input flags + NLEPosterior structured observation

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py`
- Modify: `src/gensbi/inference/nle.py`
- Test: `tests/normalizing_flows/transformer_flow/test_structured_boundary.py` (new)

**Interfaces:**
- Produces:
  - `ConditionalFlowPipeline(..., structured_obs=False, structured_cond=False)`. When a side is structured, `_squeeze_ch` is bypassed for it (the model's tokenizer/conditioner owns the reshape); `fit_standardization` computes per-element stats over axis 0 of the (unsqueezed) structured obs; `get_sampler`/`get_log_prob_fn` pass a structured single `x_o` through and broadcast it over `nsamples`.
  - `NLEPosterior(flow, prior, *, num_warmup=500, num_samples=1000, num_chains=1, structured_obs=False)`. When `structured_obs`, `potential` keeps `x_o` structured (no `atleast_1d(squeeze(...))`) and feeds `x_o[None]` to `flow.log_prob`.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_structured_boundary.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.normalizing_flows import make_tarflow
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior
from gensbi.core.prior import make_gaussian_prior

# Field NLE: obs = 4x4x1 image x, cond = 2-vector theta
H, W, Ch, D, N = 4, 4, 1, 2, 256
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = jax.random.normal(_kth, (N, D))
# x_image[:, i, j, 0] = linear(theta) + noise
_W = jax.random.normal(jax.random.PRNGKey(5), (H * W, D))
_x = (_theta @ _W.T).reshape(N, H, W, Ch) + 0.1 * jax.random.normal(_kx, (N, H, W, Ch))


def _ds_field(bs=64):
    x, theta = np.array(_x), np.array(_theta)        # (obs=image, cond=theta)
    idx = grain.MapDataset.source(list(range(N)))
    return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
            .map(lambda i: (x[np.array(i)], theta[np.array(i)])))


def _field_pipe(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), cond_dim=D, modeled="image", img_size=H,
                        patch_size=2, img_channels=Ch, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    return ConditionalFlowPipeline(flow, _ds_field(), _ds_field(),
                                   dim_obs=H * W * Ch, dim_cond=D,
                                   structured_obs=True, structured_cond=False,
                                   training_config=cfg)


def test_field_loss_finite_and_grads(tmp_path):
    pipe = _field_pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(_x[:32])           # (32, H, W, Ch)
    cond = jnp.asarray(_theta[:32])      # (32, D)
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == () and jnp.isfinite(loss)
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


def test_field_fit_standardization_image_shape(tmp_path):
    pipe = _field_pipe(tmp_path)
    pipe.fit_standardization(_x)         # (N, H, W, Ch)
    assert pipe.model.mean[...].shape == (H, W, Ch)
    assert pipe._standardized is True


def test_field_nle_potential_structured_xo(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), cond_dim=D, modeled="image", img_size=H,
                        patch_size=2, img_channels=Ch, channels=16, num_blocks=3,
                        layers_per_block=1, head_dim=8, zero_init=False)
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(flow, prior, structured_obs=True)
    x_o = jnp.zeros((H, W, Ch))
    U = post.potential(x_o)
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_structured_boundary.py -v`
Expected: FAIL (`ConditionalFlowPipeline` has no `structured_obs` kwarg; `NLEPosterior` flattens `x_o`).

- [ ] **Step 3: Write minimal implementation**

In `flow_pipeline.py`, `ConditionalFlowPipeline.__init__` — accept and store the flags:

```python
    def __init__(self, model, train_dataset, val_dataset, dim_obs, dim_cond,
                 ch_obs=1, ch_cond=1, params=None, training_config=None,
                 structured_obs=False, structured_cond=False):
        super().__init__(
            model, train_dataset, val_dataset, dim_obs, dim_cond,
            ch_obs=ch_obs, ch_cond=ch_cond, params=params,
            training_config=training_config)
        self._standardized = False
        self.structured_obs = structured_obs
        self.structured_cond = structured_cond
```

Add a helper and use it in `get_loss_fn`, `fit_standardization`, `get_sampler`, `get_log_prob_fn`:

```python
    def _prep_obs(self, x):
        return x if self.structured_obs else _squeeze_ch(x)

    def _prep_cond(self, x):
        return x if self.structured_cond else _squeeze_ch(x)
```

```python
    def get_loss_fn(self):
        def loss_fn(model, batch, key):
            obs, cond = batch
            obs = self._prep_obs(obs)
            cond = self._prep_cond(cond)
            return -jnp.mean(model.log_prob(obs, cond))
        return loss_fn
```

```python
    def fit_standardization(self, obs_data):
        obs = jnp.asarray(obs_data)
        if not self.structured_obs and obs.ndim == 3:
            obs = _squeeze_ch(obs)
        mean = jnp.mean(obs, axis=0)
        std = jnp.std(obs, axis=0)
        std = jnp.where(std < 1e-6, 1.0, std)
        self.model.set_standardization(mean, std)
        self.ema_model.set_standardization(mean, std)
        self._standardized = True
```

For `get_sampler` / `get_log_prob_fn`, branch on `structured_cond`. The structured per-example condition is just `cond.shape` after stripping a leading singleton batch; the **vector path must stay byte-identical to v1** (including `_expand_dims` on samples):

```python
    def get_sampler(self, x_o, use_ema=True):
        flow = self.ema_model if use_ema else self.model
        if self.structured_cond:
            cond = jnp.asarray(x_o)
            if cond.ndim >= 1 and cond.shape[0] == 1:
                cond = cond[0]                       # strip singleton batch

            def sampler(key, nsamples):
                cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
                return flow.sample(key, cond=cond_b)  # (nsamples, dim_obs)
            return sampler

        cond = _single_cond(x_o)                      # (dim_cond,)  [v1 path]

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples, cond.shape[0]))
            samples = flow.sample(key, cond=cond_b)
            return _expand_dims(samples)              # (nsamples, dim_obs, 1)
        return sampler

    def get_log_prob_fn(self, x_o, use_ema=True):
        flow = self.ema_model if use_ema else self.model
        if self.structured_cond:
            cond = jnp.asarray(x_o)
            if cond.ndim >= 1 and cond.shape[0] == 1:
                cond = cond[0]

            def log_prob_fn(x_1):
                obs = self._prep_obs(x_1)
                cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
                return flow.log_prob(obs, cond_b)
            return log_prob_fn

        cond = _single_cond(x_o)                      # (dim_cond,)  [v1 path]

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0], cond.shape[0]))
            return flow.log_prob(obs, cond_b)
        return log_prob_fn
```

(Structured-cond `sample` returns `(nsamples, dim_obs)` — no `_expand_dims`. For v2 image-NPE the modeled θ is a vector, so this matches `flow.sample`; Task 8 asserts `(nsamples, dim_obs)`. The vector NPE path keeps the v1 `_expand_dims` shape `(nsamples, dim_obs, 1)`, so the existing pipeline/e2e tests are unaffected.)

In `nle.py`, `NLEPosterior`:

```python
    def __init__(self, flow, prior, *, num_warmup=500, num_samples=1000,
                 num_chains=1, structured_obs=False):
        self.flow = flow
        self.prior = prior
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.structured_obs = structured_obs

    def potential(self, x_o):
        if self.structured_obs:
            x_o = jnp.asarray(x_o)
        else:
            x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))
        flow = self.flow
        prior = self.prior

        def U(theta):
            theta = jnp.asarray(theta)
            log_like = flow.log_prob(x_o[None], theta[None, :])[0]
            log_prior = prior.log_prob(theta)
            return -(log_like + log_prior)

        return U
```

(`x_o[None]` works for both: a structured `(H,W,C) -> (1,H,W,C)` and a vector `(dim,) -> (1,dim)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_structured_boundary.py -v`
Expected: PASS. Then run the existing NLE + pipeline tests to confirm the vector path is unbroken:
`JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle.py tests/normalizing_flows/transformer_flow/test_pipeline_integration.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py src/gensbi/inference/nle.py \
        tests/normalizing_flows/transformer_flow/test_structured_boundary.py
git commit -m "feat(nf): structured obs/cond pipeline flags + NLEPosterior structured x_o"
```

---

### Task 8: End-to-end smoke integration (field NLE + image NPE)

**Files:**
- Test: `tests/normalizing_flows/transformer_flow/test_structured_integration.py` (new)

**Interfaces:**
- Consumes: `make_tarflow`, `ConditionalFlowPipeline`, `NLEPosterior`, `make_gaussian_prior`. No new source — this is the CI smoke gate for both v2 setups (train 2 steps, sample/log_prob shapes + finiteness). If a test reveals a contract gap, fix it in `model.py`/`flow_pipeline.py` and note it.

- [ ] **Step 1: Write the failing test**

```python
# tests/normalizing_flows/transformer_flow/test_structured_integration.py
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

H, W, Ch, D, N = 4, 4, 1, 2, 256
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = np.array(jax.random.normal(_kth, (N, D)))
_Wm = jax.random.normal(jax.random.PRNGKey(5), (H * W, D))
_x = np.array((jnp.asarray(_theta) @ _Wm.T).reshape(N, H, W, Ch)
              + 0.1 * jax.random.normal(_kx, (N, H, W, Ch)))


def _iter(obs, cond, bs=64):
    idx = grain.MapDataset.source(list(range(len(obs))))
    return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
            .map(lambda i: (obs[np.array(i)], cond[np.array(i)])))


def test_field_nle_train_smoke_and_nuts(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), cond_dim=D, modeled="image", img_size=H,
                        patch_size=2, img_channels=Ch, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    pipe = ConditionalFlowPipeline(flow, _iter(_x, _theta), _iter(_x, _theta),
                                   dim_obs=H * W * Ch, dim_cond=D,
                                   structured_obs=True, training_config=cfg)
    pipe.fit_standardization(_x)
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)),
                        num_warmup=3, num_samples=10, structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), _x[0])
    assert s.shape == (10, D, 1) and jnp.all(jnp.isfinite(s))


def test_image_npe_train_smoke_and_sample(tmp_path):
    # NPE: obs = theta vector, cond = image
    flow = make_tarflow(nnx.Rngs(0), dim=D, modeled="vector", cond="image_prefix",
                        cond_img_size=H, cond_patch_size=2, cond_channels=Ch,
                        channels=16, num_blocks=4, layers_per_block=2, head_dim=8,
                        standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    pipe = ConditionalFlowPipeline(flow, _iter(_theta, _x), _iter(_theta, _x),
                                   dim_obs=D, dim_cond=H * W * Ch,
                                   structured_cond=True, training_config=cfg)
    pipe.fit_standardization(_theta)        # standardize the modeled theta
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    s = pipe.sample(jax.random.PRNGKey(3), _x[0:1], nsamples=16, use_ema=False)
    assert s.shape == (16, D) and jnp.all(jnp.isfinite(s))
    lp = pipe.log_prob(_theta[:5], _x[0:1], use_ema=False)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))
```

- [ ] **Step 2: Run test to verify it fails (or is the verification gate)**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_structured_integration.py -v`
Expected: FAIL only if a contract gap exists (shape mismatch in the structured sampler/log_prob, or standardization shape). Otherwise this is the gate.

- [ ] **Step 3: Fix any contract gaps (if needed)**

If a test errors on a shape/contract mismatch, the minimal fix lives in `model.py` (e.g. `sample` broadcasting of a structured `cond`) or `flow_pipeline.py` (the structured `get_sampler`/`get_log_prob_fn`). Most likely none is needed — `flow.sample(key, cond=image_batch)` sets `nsamples = cond.shape[0]` and returns `(B, dim_obs)`, and the field-NLE path reuses the vector `NLEPosterior.sample` (NUTS over θ).

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow/test_structured_integration.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/normalizing_flows/transformer_flow/test_structured_integration.py \
        src/gensbi/normalizing_flows/transformer_flow/model.py \
        src/gensbi/recipes/flow_pipeline.py
git commit -m "test(nf): e2e smoke for field-NLE and image-NPE TransformerFlow"
```

---

### Task 9: GPU recovery scripts (field NLE + image NPE)

**Files:**
- Create: `scripts/tarflow_field_nle_recovery.py`
- Create: `scripts/tarflow_image_npe_recovery.py`

**Interfaces:**
- Standalone scripts mirroring `scripts/tarflow_nle_recovery.py`: an `argparse` CLI with `--smoke` (minimal wiring check, no recovery assertion) and a full mode (recovery assertions, intended for GPU). Not wired into pytest — the CI smoke gate is Task 8.

- [ ] **Step 1: Create `scripts/tarflow_field_nle_recovery.py`**

```python
"""Field-level NLE recovery for TransformerFlow (image modeled x, vector theta).
Standalone; intended for cluster/GPU scheduling, not the pytest battery.

Linear-Gaussian: x_image = (G @ theta) reshaped to (H,W,1) + sigma*noise, with a
known G so the posterior over theta given the full image is analytic. NLE+NUTS
should recover it.
"""
import argparse
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description="Field-NLE TransformerFlow recovery.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-warmup", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", type=str, default=None)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--atol", type=float, default=0.25)
    args = p.parse_args()

    if args.platform is not None:
        import os
        os.environ["JAX_PLATFORMS"] = args.platform

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grain
    from flax import nnx
    from gensbi.core.prior import make_gaussian_prior
    from gensbi.normalizing_flows import make_tarflow
    from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
    from gensbi.inference import NLEPosterior

    smoke = args.smoke
    n_data = args.n_data or (2_000 if smoke else 40_000)
    nsteps = args.steps or (10 if smoke else 6_000)
    num_warmup = args.num_warmup or (5 if smoke else 500)
    num_samples = args.num_samples or (20 if smoke else 4_000)
    val_every = 1 if smoke else 200

    H = Wd = 4
    Ch, D, SIGMA = 1, 2, 0.5
    Mdim = H * Wd
    G = jax.random.normal(jax.random.PRNGKey(123), (Mdim, D))   # (16, 2)

    def simulate(key, n):
        kth, ke = jax.random.split(key)
        theta = jax.random.normal(kth, (n, D))
        flat = theta @ G.T + SIGMA * jax.random.normal(ke, (n, Mdim))
        return theta, flat.reshape(n, H, Wd, Ch)

    def analytic_posterior(x_o_flat):
        prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
        cov = jnp.linalg.inv(prec)
        mean = cov @ (G.T @ x_o_flat) / SIGMA ** 2
        return mean, cov

    t0 = time.time()
    theta, x = simulate(jax.random.PRNGKey(args.seed), n_data)
    n_train = int(n_data * 0.9)

    def make_ds(obs, cond):
        idx = grain.MapDataset.source(list(range(len(obs))))
        obs_n, cond_n = np.array(obs), np.array(cond)
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(256)
                .map(lambda i: (obs_n[np.array(i)], cond_n[np.array(i)])))

    flow = make_tarflow(nnx.Rngs(args.seed), cond_dim=D, modeled="image",
                        img_size=H, patch_size=2, img_channels=Ch,
                        channels=args.channels, num_blocks=args.num_blocks,
                        layers_per_block=2, head_dim=args.head_dim,
                        standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=nsteps, val_every=val_every, max_lr=3e-4,
                    checkpoint_dir=tempfile.mkdtemp(), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(x[:n_train], theta[:n_train]),
                                   make_ds(x[n_train:], theta[n_train:]),
                                   dim_obs=Mdim, dim_cond=D, structured_obs=True,
                                   training_config=cfg)
    pipe.fit_standardization(x[:n_train])
    pipe.train(nnx.Rngs(args.seed), nsteps=nsteps, save_model=False)

    theta_o = jnp.array([0.7, -0.4])
    x_o = (theta_o @ G.T).reshape(H, Wd, Ch)
    mean_a, cov_a = analytic_posterior(x_o.reshape(-1))
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)),
                        num_warmup=num_warmup, num_samples=num_samples,
                        structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]
    mean_s, cov_s = jnp.mean(s, axis=0), jnp.cov(s.T)
    print(f"mode={'SMOKE' if smoke else 'FULL'} elapsed={time.time()-t0:.1f}s")
    print(f"analytic mean {mean_a}  achieved {mean_s}")
    print(f"analytic cov\n{cov_a}\nachieved\n{cov_s}")

    if smoke:
        assert jnp.all(jnp.isfinite(mean_s)) and jnp.all(jnp.isfinite(cov_s))
        print("SMOKE OK"); sys.exit(0)
    ok = bool(jnp.allclose(mean_s, mean_a, atol=args.atol)
              and jnp.allclose(cov_s, cov_a, atol=args.atol))
    print("RECOVERY PASS" if ok else "RECOVERY FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it**

Run: `JAX_PLATFORMS=cpu .venv/bin/python scripts/tarflow_field_nle_recovery.py --smoke`
Expected: prints `SMOKE OK`, exit 0.

- [ ] **Step 3: Create `scripts/tarflow_image_npe_recovery.py`**

```python
"""Image-conditioned NPE recovery for TransformerFlow (vector theta, image cond).
Standalone; intended for cluster/GPU scheduling, not the pytest battery.

Linear-Gaussian: image x = (G @ theta) reshaped + sigma*noise. Train q(theta|x)
and recover the analytic posterior by direct sampling (no NUTS).
"""
import argparse
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description="Image-NPE TransformerFlow recovery.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", type=str, default=None)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--atol", type=float, default=0.25)
    args = p.parse_args()

    if args.platform is not None:
        import os
        os.environ["JAX_PLATFORMS"] = args.platform

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grain
    from flax import nnx
    from gensbi.normalizing_flows import make_tarflow
    from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline

    smoke = args.smoke
    n_data = args.n_data or (2_000 if smoke else 40_000)
    nsteps = args.steps or (10 if smoke else 6_000)
    num_samples = args.num_samples or (200 if smoke else 4_000)
    val_every = 1 if smoke else 200

    H = Wd = 4
    Ch, D, SIGMA = 1, 2, 0.5
    Mdim = H * Wd
    G = jax.random.normal(jax.random.PRNGKey(123), (Mdim, D))

    def simulate(key, n):
        kth, ke = jax.random.split(key)
        theta = jax.random.normal(kth, (n, D))
        flat = theta @ G.T + SIGMA * jax.random.normal(ke, (n, Mdim))
        return theta, flat.reshape(n, H, Wd, Ch)

    def analytic_posterior(x_o_flat):
        prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
        cov = jnp.linalg.inv(prec)
        mean = cov @ (G.T @ x_o_flat) / SIGMA ** 2
        return mean, cov

    t0 = time.time()
    theta, x = simulate(jax.random.PRNGKey(args.seed), n_data)
    n_train = int(n_data * 0.9)

    def make_ds(obs, cond):
        idx = grain.MapDataset.source(list(range(len(obs))))
        obs_n, cond_n = np.array(obs), np.array(cond)
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(256)
                .map(lambda i: (obs_n[np.array(i)], cond_n[np.array(i)])))

    flow = make_tarflow(nnx.Rngs(args.seed), dim=D, modeled="vector",
                        cond="image_prefix", cond_img_size=H, cond_patch_size=2,
                        cond_channels=Ch, channels=args.channels,
                        num_blocks=args.num_blocks, layers_per_block=2,
                        head_dim=args.head_dim, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=nsteps, val_every=val_every, max_lr=3e-4,
                    checkpoint_dir=tempfile.mkdtemp(), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(theta[:n_train], x[:n_train]),
                                   make_ds(theta[n_train:], x[n_train:]),
                                   dim_obs=D, dim_cond=Mdim, structured_cond=True,
                                   training_config=cfg)
    pipe.fit_standardization(theta[:n_train])
    pipe.train(nnx.Rngs(args.seed), nsteps=nsteps, save_model=False)

    theta_o = jnp.array([0.7, -0.4])
    x_o = (theta_o @ G.T).reshape(1, H, Wd, Ch)
    mean_a, cov_a = analytic_posterior(x_o.reshape(-1))
    s = pipe.sample(jax.random.PRNGKey(7), x_o, nsamples=num_samples)
    mean_s, cov_s = jnp.mean(s, axis=0), jnp.cov(s.T)
    print(f"mode={'SMOKE' if smoke else 'FULL'} elapsed={time.time()-t0:.1f}s")
    print(f"analytic mean {mean_a}  achieved {mean_s}")
    print(f"analytic cov\n{cov_a}\nachieved\n{cov_s}")

    if smoke:
        assert jnp.all(jnp.isfinite(mean_s)) and jnp.all(jnp.isfinite(cov_s))
        print("SMOKE OK"); sys.exit(0)
    ok = bool(jnp.allclose(mean_s, mean_a, atol=args.atol)
              and jnp.allclose(cov_s, cov_a, atol=args.atol))
    print("RECOVERY PASS" if ok else "RECOVERY FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-run it**

Run: `JAX_PLATFORMS=cpu .venv/bin/python scripts/tarflow_image_npe_recovery.py --smoke`
Expected: prints `SMOKE OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/tarflow_field_nle_recovery.py scripts/tarflow_image_npe_recovery.py
git commit -m "test(nf): GPU recovery scripts for field-NLE and image-NPE (smoke + full)"
```

---

## Final verification

Run the full transformer-flow suite and the whole NF package to confirm no regressions:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/transformer_flow -v
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows -q
JAX_PLATFORMS=cpu .venv/bin/python scripts/tarflow_field_nle_recovery.py --smoke
JAX_PLATFORMS=cpu .venv/bin/python scripts/tarflow_image_npe_recovery.py --smoke
```

Expected: all fast tests pass (v1 + v2); both smoke scripts print `SMOKE OK`. The full recoveries (`--platform gpu`, no `--smoke`) are run by the user on GPU.
