# FieldDiT Phase 1 — Model & Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `FieldDiT` model and its `FieldDiTParams` config — a conditional flow-matching network that maps a noisy 2D field + time + a (global/statistical) condition to a pixel-space velocity field — and verify it produces outputs of the expected nature (correct shape, finite, identity-at-init, differentiable).

**Architecture:** A SiD2-style residual conv U-Net (`ObsEncoder` ↓ / `ObsDecoder` ↑) whose bottleneck is upgraded from self-attention to the Flux1 MMDiT core. The condition is embedded into a token stream (`ScalarCondEmbedder`) and merged with the obs tokens via joint attention; a pooled condition summary modulates the decoder (flagged-C). Conv blocks use AdaGN-zero modulation; the transformer reuses Flux1's AdaLN-zero. SiD2 residual skips (subtract/add, not concat) bypass the transformer in conv space. The final decoder conv is zero-initialized, so the velocity field is exactly zero at init.

**Tech Stack:** JAX + `flax.nnx`. Reuses Flux1 layers (`DoubleStreamBlock`, `SingleStreamBlock`, `EmbedND`, `MLPEmbedder`, `timestep_embedding`, `FeatureEmbedder`) and recipe utils (`patchify_2d`, `init_ids_2d`, `init_ids_1d`). New code lives under `src/gensbi/experimental/models/fielddit/`.

---

## Scope (read before starting)

**In scope (this plan):** the `FieldDiT` model + `FieldDiTParams`, built bottom-up as small, independently-testable `flax.nnx` modules, with tests that each component and the assembled model produce outputs of the expected *nature* (shape, dtype, finiteness, identity-at-init, differentiability).

**Explicitly OUT of scope (deferred to later plans):**
- No training pipeline / no `FieldConditionalPipeline` wiring (the model is callable and differentiable; wiring into a `GenerativeMethod`/`ConditionalPipeline` comes later).
- No GRF 256² validation, power-spectrum recovery, or field-space SBC/TARP (a separate experimental plan).
- No 1D path. The 2D component interfaces are designed so a 1D variant can be added later, but it is not built here.
- No Phase 2 image / spatially-aligned (Kontext) conditioning.
- No classifier-free-guidance logic beyond a minimal optional `guidance_embed` plumbing hook.

**Spec:** `docs/superpowers/specs/2026-06-09-fielddit-design.md` (§3 architecture, §3.4 skips, §3.5 config). This plan implements the Phase-1 architecture only (not §6 validation).

## Environment / how to run tests

Use **either** `uv run <cmd>` **or** `mamba activate gensbi` then `<cmd>` — they are equivalent and both have all dependencies. All commands below use `uv run`. Tests force CPU via `JAX_PLATFORMS=cpu` (set in `pyproject.toml` `[tool.pytest.ini_options]`), and pytest runs with `-n 2` (xdist) by default. New tests under `tests/` are auto-collected (`testpaths = ["tests"]`).

## Conventions used throughout

- **Tensor layout:** fields are NHWC `(B, H, W, C)`; token streams are `(B, num_tokens, hidden_size)`; the modulation vector `vec` is `(B, hidden_size)`.
- **`param_dtype`:** modules default to `jnp.bfloat16` (matching the repo). **All tests pass `param_dtype=jnp.float32`** for exact numeric checks.
- **`vec_dim == hidden_size`:** the same `vec` modulates both conv blocks (AdaGN-zero) and transformer blocks (AdaLN-zero), so it is sized at `hidden_size`.
- **`hidden_size` is derived:** `hidden_size = sum(axes_dim) * num_heads` (mirrors `Flux1Params`). `head_dim = hidden_size // num_heads = sum(axes_dim)`.
- **`nnx` submodule containers:** this flax version does **not** track bare Python lists assigned as attributes (it raises, demanding `nnx.data`/`nnx.List`). Use `nnx.Sequential(*modules)` for block lists and iterate `.layers`; use a bare `nnx.Module()` instance with attributes for a "stage" struct (the pattern already used in `autoencoder_2d.Encoder2D`). **Do not** assign a plain `list` of modules to `self.x`.
- **`depatchify_2d` fix:** `gensbi.recipes.utils.depatchify_2d` currently cannot infer `(h, w)` from token count alone (it raises an `EinopsError` even for square grids) and has **zero callers** in the codebase. Task 5 extends it with an optional `grid=(h, w)` argument so it can actually invert `patchify_2d`; the `Untokenizer` passes the known token grid. This fixes a genuinely broken utility rather than working around it.

## File structure

```
src/gensbi/experimental/models/fielddit/
├── __init__.py     # exports FieldDiT, FieldDiTParams
├── blocks.py       # _safe_groups, ConvModulation (AdaGN-zero), ModulatedResBlock2D
├── codec.py        # Downsample2D, Upsample2D, ObsEncoder, ObsDecoder, Tokenizer, Untokenizer
├── cond.py         # ScalarCondEmbedder (cond tokens + pooled summary)
├── core.py         # MMDiTCore (Flux1 double/single-stream + rope ids + absolute cond ids)
└── model.py        # FieldDiTParams (dataclass), FieldDiT (assembly + forward)

tests/experimental/models/fielddit/
├── test_blocks.py
├── test_codec.py
├── test_cond.py
├── test_core.py
└── test_model.py
```

Each file has one clear responsibility: `blocks.py` = the modulated conv primitive; `codec.py` = the conv U-Net halves + patch boundary; `cond.py` = condition embedding; `core.py` = the transformer bottleneck; `model.py` = config + assembly.

---

## Task 1: `blocks.py` — `ConvModulation` + `ModulatedResBlock2D`

The AdaGN-zero conv primitive. `ConvModulation` projects `vec → (scale, shift, gate)` broadcast over space, zero-initialized (neutral modulation, closed gate at init). `ModulatedResBlock2D` is the SiD2 residual block with FiLM over GroupNorm and a zero-init gate so the block is identity at init: `out = residual + gate · h`.

**Files:**
- Create: `src/gensbi/experimental/models/fielddit/__init__.py`
- Create: `src/gensbi/experimental/models/fielddit/blocks.py`
- Test: `tests/experimental/models/fielddit/test_blocks.py`

- [ ] **Step 1: Create the empty package `__init__.py`**

Create `src/gensbi/experimental/models/fielddit/__init__.py` with a placeholder (real exports added in Task 10):

```python
"""FieldDiT: conditional flow-matching for 2D field-level inference (Phase 1)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/experimental/models/fielddit/test_blocks.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.blocks import (
    _safe_groups,
    ConvModulation,
    ModulatedResBlock2D,
)


def test_safe_groups_divides():
    assert _safe_groups(8, 4) == 4
    assert _safe_groups(8, 32) == 8  # cannot exceed num_features
    assert _safe_groups(7, 8) == 1   # odd channels fall back to 1 group


def test_conv_modulation_zero_init():
    mod = ConvModulation(vec_dim=16, channels=8, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    scale, shift, gate = mod(jnp.ones((2, 16)))
    assert scale.shape == (2, 1, 1, 8)
    assert shift.shape == (2, 1, 1, 8)
    assert gate.shape == (2, 1, 1, 8)
    # zero-initialized linear => neutral modulation at init
    assert jnp.allclose(scale, 0.0)
    assert jnp.allclose(shift, 0.0)
    assert jnp.allclose(gate, 0.0)


def test_modulated_resblock_identity_at_init():
    block = ModulatedResBlock2D(
        in_channels=8, out_channels=8, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 12, 8))
    vec = jnp.ones((2, 16))
    out = block(x, vec)
    assert out.shape == (2, 12, 12, 8)
    # gate is zero at init => out == residual == x (in_channels == out_channels)
    assert jnp.allclose(out, x, atol=1e-5)


def test_modulated_resblock_channel_change():
    block = ModulatedResBlock2D(
        in_channels=8, out_channels=16, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 12, 8))
    out = block(x, jnp.ones((2, 16)))
    assert out.shape == (2, 12, 12, 16)
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_blocks.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `blocks` module yet).

- [ ] **Step 4: Write minimal implementation**

Create `src/gensbi/experimental/models/fielddit/blocks.py`:

```python
"""AdaGN-zero modulated residual conv block for the FieldDiT conv codec.

Ported (not imported) from the reference SiD2 ``ResidualBlock2D`` (Keras) and
GenSBI's ``ResnetBlock2D``: FiLM scale/shift over GroupNorm, plus a
conditioning-predicted *gate* (zero-initialized) so each block is identity at
initialization (``out = residual + gate * h``).
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike


def _safe_groups(num_features: int, groups: int) -> int:
    """Largest group count <= ``groups`` that divides ``num_features``.

    GenSBI's conv blocks hardcode ``num_groups=32``, which breaks on the small
    channel widths used here. ``gcd`` always yields a valid divisor (>= 1).
    """
    return math.gcd(int(num_features), int(groups))


class ConvModulation(nnx.Module):
    """Project a global ``vec`` to (scale, shift, gate) for an NHWC feature map.

    Mirrors Flux1's ``Modulation`` (zero-init linear => neutral modulation /
    closed gate at init), but reshapes outputs to ``(B, 1, 1, C)`` so they
    broadcast over the spatial dims.
    """

    def __init__(
        self,
        vec_dim: int,
        channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.channels = channels
        self.lin = nnx.Linear(
            in_features=vec_dim,
            out_features=3 * channels,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, vec):
        out = self.lin(nnx.silu(vec))
        scale, shift, gate = jnp.split(out, 3, axis=-1)
        reshape = lambda z: z[:, None, None, :]
        return reshape(scale), reshape(shift), reshape(gate)


class ModulatedResBlock2D(nnx.Module):
    """SiD2-style residual conv block with AdaGN-zero modulation (NHWC).

    Structure: ``norm1 -> silu -> conv1 -> norm2 -> FiLM(scale,shift) -> silu
    -> conv2``, returned as ``residual + gate * h``. ``norm2`` has its own
    affine disabled (the predicted scale/shift is the sole affine). The gate is
    zero-initialized, giving (a) identity at init and (b) condition-dependent
    block strength.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vec_dim: int,
        norm_groups: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=_safe_groups(in_channels, norm_groups),
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        # affine off: ConvModulation provides the sole scale/shift
        self.norm2 = nnx.GroupNorm(
            num_groups=_safe_groups(out_channels, norm_groups),
            num_features=out_channels,
            epsilon=1e-6,
            use_scale=False,
            use_bias=False,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.mod = ConvModulation(
            vec_dim=vec_dim, channels=out_channels, rngs=rngs, param_dtype=param_dtype
        )
        if in_channels != out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                rngs=rngs,
                param_dtype=param_dtype,
            )
        else:
            self.nin_shortcut = None

    def __call__(self, x, vec):
        residual = x if self.nin_shortcut is None else self.nin_shortcut(x)
        h = self.conv1(nnx.silu(self.norm1(x)))
        scale, shift, gate = self.mod(vec)
        h = self.norm2(h)
        h = (1 + scale) * h + shift
        h = self.conv2(nnx.silu(h))
        return residual + gate * h
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_blocks.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/__init__.py \
        src/gensbi/experimental/models/fielddit/blocks.py \
        tests/experimental/models/fielddit/test_blocks.py
git commit -m "feat(fielddit): AdaGN-zero modulated conv resblock"
```

---

## Task 2: `codec.py` — `Downsample2D` + `Upsample2D` (channel-changing)

Stride-2 conv downsample and nearest-2× upsample that also change channel count (the reference UpSample/DownSample project to the next stage width). These are distinct from `autoencoder_2d`'s versions, which keep channels fixed.

**Files:**
- Create: `src/gensbi/experimental/models/fielddit/codec.py`
- Test: `tests/experimental/models/fielddit/test_codec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/experimental/models/fielddit/test_codec.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.codec import Downsample2D, Upsample2D


def test_downsample_halves_and_changes_channels():
    down = Downsample2D(in_channels=8, out_channels=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 8))
    out = down(x)
    assert out.shape == (2, 8, 8, 16)


def test_upsample_doubles_and_changes_channels():
    up = Upsample2D(in_channels=16, out_channels=8, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 16))
    out = up(x)
    assert out.shape == (2, 16, 16, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py -v`
Expected: FAIL with `ImportError` (no `codec` module yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/gensbi/experimental/models/fielddit/codec.py` with the imports and these two classes (more classes are appended in Tasks 3–5):

```python
"""Conv U-Net halves and the patch boundary for FieldDiT.

ObsEncoder (conv down, time-only modulation, captures SiD2 skips) and
ObsDecoder (conv up, residual skips, time+cond modulation, zero-init final
conv) sandwich the MMDiT bottleneck; Tokenizer/Untokenizer cross the
patchify boundary.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.recipes.utils import patchify_2d, depatchify_2d
from gensbi.experimental.models.fielddit.blocks import (
    ModulatedResBlock2D,
    _safe_groups,
)


class Downsample2D(nnx.Module):
    """Stride-2 conv that also changes channel count (asymmetric pad, AE-style)."""

    def __init__(self, in_channels, out_channels, rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x):
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
        return self.conv(x)


class Upsample2D(nnx.Module):
    """Nearest-neighbour 2x upsample + conv that also changes channel count."""

    def __init__(self, in_channels, out_channels, rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        return self.conv(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py \
        tests/experimental/models/fielddit/test_codec.py
git commit -m "feat(fielddit): channel-changing down/upsample blocks"
```

---

## Task 3: `codec.py` — `ObsEncoder`

Conv encoder: `conv_in` to base width, then `D = len(widths) - 1` stages, each `res_blocks` `ModulatedResBlock2D` (modulated by **time only**) followed by a channel-changing downsample. Captures `pos_skips` (pre-downsample) and `neg_skips` (post-downsample) per stage. Returns the bottleneck feature (`== neg_skips[-1]`).

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/codec.py` (append `ObsEncoder`)
- Test: `tests/experimental/models/fielddit/test_codec.py` (append)

- [ ] **Step 1: Write the failing test (append to `test_codec.py`)**

```python
from gensbi.experimental.models.fielddit.codec import ObsEncoder


def test_obs_encoder_shapes_and_skips():
    widths = (8, 16, 32)  # D = 2 downsamples
    enc = ObsEncoder(
        in_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 1))
    time_vec = jnp.ones((2, 16))
    feat, pos_skips, neg_skips = enc(x, time_vec)

    # bottleneck: 32 -> 16 -> 8, width 32
    assert feat.shape == (2, 8, 8, 32)
    assert len(pos_skips) == 2 and len(neg_skips) == 2
    # pos_skips captured pre-downsample at stage widths/resolutions
    assert pos_skips[0].shape == (2, 32, 32, 8)
    assert pos_skips[1].shape == (2, 16, 16, 16)
    # neg_skips captured post-downsample
    assert neg_skips[0].shape == (2, 16, 16, 16)
    assert neg_skips[1].shape == (2, 8, 8, 32)
    # the returned feature is exactly the last neg_skip
    assert jnp.allclose(feat, neg_skips[-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py::test_obs_encoder_shapes_and_skips -v`
Expected: FAIL with `ImportError` (`ObsEncoder` not defined).

- [ ] **Step 3: Write minimal implementation (append to `codec.py`)**

```python
class ObsEncoder(nnx.Module):
    """Conv encoder: down-sampling stages with time-only AdaGN-zero modulation.

    ``widths`` has length ``D + 1`` (one width per resolution incl. the
    bottleneck). Stage ``j`` (j = 0..D-1) runs ``res_blocks`` blocks at width
    ``widths[j]``, then downsamples ``widths[j] -> widths[j+1]``. Returns the
    bottleneck feature plus per-stage ``pos_skips`` (pre-downsample) and
    ``neg_skips`` (post-downsample) for the SiD2 residual decoder.
    """

    def __init__(
        self,
        in_channels,
        widths,
        res_blocks,
        vec_dim,
        norm_groups,
        rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.widths = tuple(widths)
        self.depth = len(self.widths) - 1
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.widths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.down = nnx.Sequential()
        for j in range(self.depth):
            stage = nnx.Module()
            stage.block = nnx.Sequential(
                *[
                    ModulatedResBlock2D(
                        self.widths[j], self.widths[j], vec_dim, norm_groups,
                        rngs=rngs, param_dtype=param_dtype,
                    )
                    for _ in range(res_blocks)
                ]
            )
            stage.downsample = Downsample2D(
                self.widths[j], self.widths[j + 1], rngs=rngs, param_dtype=param_dtype
            )
            self.down.layers.append(stage)

    def __call__(self, x, time_vec):
        h = self.conv_in(x)
        pos_skips = []
        neg_skips = []
        for j in range(self.depth):
            stage = self.down.layers[j]
            for blk in stage.block.layers:
                h = blk(h, time_vec)
            pos_skips.append(h)
            h = stage.downsample(h)
            neg_skips.append(h)
        return h, pos_skips, neg_skips
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py -v`
Expected: PASS (all codec tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py \
        tests/experimental/models/fielddit/test_codec.py
git commit -m "feat(fielddit): ObsEncoder with time-only modulation and SiD2 skips"
```

---

## Task 4: `codec.py` — `ObsDecoder`

Mirror of `ObsEncoder`. Per stage (deepest→shallowest): `h = h - neg_skips[j]`; `upsample(widths[j+1]->widths[j])`; `h = h + pos_skips[j]`; `res_blocks` blocks at `widths[j]` (modulated by time+cond `vec`). Final `norm_out` + **zero-init** `conv_out` → velocity field. Zero-init makes the output exactly zero at init.

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/codec.py` (append `ObsDecoder`)
- Test: `tests/experimental/models/fielddit/test_codec.py` (append)

- [ ] **Step 1: Write the failing test (append to `test_codec.py`)**

```python
from gensbi.experimental.models.fielddit.codec import ObsDecoder


def test_obs_decoder_reconstructs_field_shape_and_zero_init():
    widths = (8, 16, 32)
    enc = ObsEncoder(
        in_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )
    dec = ObsDecoder(
        in_channels=1, widths=widths, res_blocks=2, vec_dim=16, norm_groups=4,
        rngs=nnx.Rngs(1), param_dtype=jnp.float32,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 1))
    vec = jnp.ones((2, 16))
    feat, pos_skips, neg_skips = enc(x, vec)
    out = dec(feat, vec, pos_skips, neg_skips)

    assert out.shape == (2, 32, 32, 1)
    # zero-init final conv => exactly-zero velocity at init (also catches any
    # upstream NaN, since 0 * NaN == NaN would break this).
    assert jnp.allclose(out, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py::test_obs_decoder_reconstructs_field_shape_and_zero_init -v`
Expected: FAIL with `ImportError` (`ObsDecoder` not defined).

- [ ] **Step 3: Write minimal implementation (append to `codec.py`)**

```python
class ObsDecoder(nnx.Module):
    """Conv decoder: SiD2 residual skips + time+cond AdaGN-zero modulation.

    Mirrors ``ObsEncoder``. ``self.up.layers[i]`` corresponds to encoder stage
    ``j = depth - 1 - i``. Per stage: subtract the matching ``neg_skip``,
    upsample ``widths[j+1] -> widths[j]``, add the matching ``pos_skip``, then
    run the stage's blocks. The final conv is zero-initialized so the velocity
    field is exactly zero at initialization.
    """

    def __init__(
        self,
        in_channels,
        widths,
        res_blocks,
        vec_dim,
        norm_groups,
        rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.widths = tuple(widths)
        self.depth = len(self.widths) - 1
        self.up = nnx.Sequential()
        for j in reversed(range(self.depth)):
            stage = nnx.Module()
            stage.upsample = Upsample2D(
                self.widths[j + 1], self.widths[j], rngs=rngs, param_dtype=param_dtype
            )
            stage.block = nnx.Sequential(
                *[
                    ModulatedResBlock2D(
                        self.widths[j], self.widths[j], vec_dim, norm_groups,
                        rngs=rngs, param_dtype=param_dtype,
                    )
                    for _ in range(res_blocks)
                ]
            )
            self.up.layers.append(stage)

        self.norm_out = nnx.GroupNorm(
            num_groups=_safe_groups(self.widths[0], norm_groups),
            num_features=self.widths[0],
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=self.widths[0],
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, feat, vec, pos_skips, neg_skips):
        h = feat
        for i in range(self.depth):
            j = self.depth - 1 - i
            stage = self.up.layers[i]
            h = h - neg_skips[j]
            h = stage.upsample(h)
            h = h + pos_skips[j]
            for blk in stage.block.layers:
                h = blk(h, vec)
        h = nnx.silu(self.norm_out(h))
        return self.conv_out(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py \
        tests/experimental/models/fielddit/test_codec.py
git commit -m "feat(fielddit): ObsDecoder with residual skips and zero-init output"
```

---

## Task 5: `depatchify_2d` fix + `codec.py` `Tokenizer`/`Untokenizer`

The patch boundary. First fix `depatchify_2d` (zero callers today, broken: it cannot infer `(h, w)` from token count) to accept an optional `grid`. Then build `Tokenizer` (`patchify_2d` + `Linear(C·p² -> hidden)`) and `Untokenizer` (`Linear(hidden -> C·p²)` + `depatchify_2d(grid=...)`).

**Files:**
- Modify: `src/gensbi/recipes/utils.py:86-88` (`depatchify_2d`)
- Modify: `tests/recipes/test_pipeline_utils.py` (append a round-trip test; it already imports `patchify_2d`)
- Modify: `src/gensbi/experimental/models/fielddit/codec.py` (append `Tokenizer`, `Untokenizer`)
- Test: `tests/experimental/models/fielddit/test_codec.py` (append)

### Part A — fix `depatchify_2d`

- [ ] **Step 1: Write the failing test (append to `tests/recipes/test_pipeline_utils.py`)**

Add `depatchify_2d` to the existing `from gensbi.recipes.utils import (patchify_2d, ...)` import group, then add:

```python
def test_depatchify_2d_roundtrip_square():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 3))
    patched = patchify_2d(x, size=2)            # (2, 64, 12)
    restored = depatchify_2d(patched, size=2, grid=(8, 8))
    assert restored.shape == x.shape
    assert jnp.allclose(restored, x)


def test_depatchify_2d_roundtrip_nonsquare():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 8, 3))
    patched = patchify_2d(x, size=2)            # (2, 32, 12)
    restored = depatchify_2d(patched, size=2, grid=(8, 4))
    assert restored.shape == x.shape
    assert jnp.allclose(restored, x)


def test_depatchify_2d_infers_square_when_grid_omitted():
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 16, 16, 3))
    patched = patchify_2d(x, size=2)
    restored = depatchify_2d(patched, size=2)   # grid=None => assume square
    assert restored.shape == x.shape
    assert jnp.allclose(restored, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recipes/test_pipeline_utils.py -k depatchify -v`
Expected: FAIL (current `depatchify_2d` takes no `grid` arg → `TypeError`, and the no-grid case raises `EinopsError`).

- [ ] **Step 3: Replace `depatchify_2d` in `src/gensbi/recipes/utils.py`**

Replace the existing definition:

```python
@jax.jit(static_argnames=["size"])
def depatchify_2d(x: Array, size=2):
    return rearrange(x, "b (h w) (c ph pw) -> b (h ph) (w pw) c", ph=size, pw=size)
```

with:

```python
@jax.jit(static_argnames=["size", "grid"])
def depatchify_2d(x: Array, size=2, grid=None):
    """Inverse of :func:`patchify_2d`.

    Parameters
    ----------
    x : Array
        Patchified tensor of shape ``(B, h*w, C*size*size)``.
    size : int
        Patch edge length used by :func:`patchify_2d`.
    grid : tuple of int, optional
        The ``(h, w)`` patch grid. The grid cannot be inferred from the token
        count alone, so it is required for non-square grids. If ``None``, a
        square grid (``h == w``) is assumed.
    """
    if grid is None:
        n = x.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"Cannot infer a square grid from {n} tokens; pass grid=(h, w)."
            )
        h = w = side
    else:
        h, w = grid
    return rearrange(
        x, "b (h w) (c ph pw) -> b (h ph) (w pw) c", h=h, w=w, ph=size, pw=size
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recipes/test_pipeline_utils.py -k depatchify -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/utils.py tests/recipes/test_pipeline_utils.py
git commit -m "fix(recipes): make depatchify_2d invert patchify_2d via explicit grid"
```

### Part B — `Tokenizer` / `Untokenizer`

- [ ] **Step 6: Write the failing test (append to `test_codec.py`)**

```python
from gensbi.experimental.models.fielddit.codec import Tokenizer, Untokenizer


def test_tokenizer_untokenizer_roundtrip_shape():
    c_bottleneck, p, hidden = 32, 2, 16
    tok = Tokenizer(c_bottleneck, p, hidden, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    untok = Untokenizer(c_bottleneck, p, hidden, rngs=nnx.Rngs(1), param_dtype=jnp.float32)

    feat = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, c_bottleneck))
    tokens = tok(feat)
    # token grid = (8/2, 8/2) = (4, 4) => 16 tokens, hidden=16
    assert tokens.shape == (2, 16, hidden)

    back = untok(tokens, grid=(4, 4))
    # shape round-trip (values differ; the two Linears are not inverses)
    assert back.shape == feat.shape
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py::test_tokenizer_untokenizer_roundtrip_shape -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 8: Write minimal implementation (append to `codec.py`)**

```python
class Tokenizer(nnx.Module):
    """Patchify a conv feature map and project to ``hidden_size`` tokens."""

    def __init__(self, in_channels, patch_size, hidden_size, rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.patch_size = patch_size
        self.proj = nnx.Linear(
            in_features=in_channels * patch_size * patch_size,
            out_features=hidden_size,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, feat):
        x = patchify_2d(feat, size=self.patch_size)  # (B, N, C * p * p)
        return self.proj(x)                          # (B, N, hidden)


class Untokenizer(nnx.Module):
    """Project tokens back to patch pixels and depatchify to a conv feature map.

    ``grid`` is the ``(h, w)`` token grid (``feat_h // p``, ``feat_w // p``),
    passed to the (now grid-aware) ``depatchify_2d``.
    """

    def __init__(self, out_channels, patch_size, hidden_size, rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nnx.Linear(
            in_features=hidden_size,
            out_features=out_channels * patch_size * patch_size,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, tokens, grid):
        x = self.proj(tokens)  # (B, N, C * p * p)
        return depatchify_2d(x, size=self.patch_size, grid=tuple(grid))
```

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_codec.py -v`
Expected: PASS (all codec tests).

- [ ] **Step 10: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/codec.py \
        tests/experimental/models/fielddit/test_codec.py
git commit -m "feat(fielddit): patch-boundary Tokenizer/Untokenizer"
```

---

## Task 6: `cond.py` — `ScalarCondEmbedder`

The Phase-1 condition embedder for global/statistical conditions (θ scalars / a small feature vector). Projects each condition token to `hidden_size`, and produces a pooled summary used for flagged-C decoder modulation.

**Files:**
- Create: `src/gensbi/experimental/models/fielddit/cond.py`
- Test: `tests/experimental/models/fielddit/test_cond.py`

- [ ] **Step 1: Write the failing test**

Create `tests/experimental/models/fielddit/test_cond.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.experimental.models.fielddit.cond import ScalarCondEmbedder


def test_scalar_cond_embedder_tokens_and_summary():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 1))  # (B, k=3, c=1)
    tokens, summary = emb(cond)
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
    assert jnp.all(jnp.isfinite(tokens)) and jnp.all(jnp.isfinite(summary))


def test_scalar_cond_embedder_accepts_2d_input():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(0), (2, 3))  # (B, k) -> expanded to (B, k, 1)
    tokens, summary = emb(cond)
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_cond.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/gensbi/experimental/models/fielddit/cond.py`:

```python
"""Pluggable condition embedders for FieldDiT (Phase 1: scalar / vector).

A condition is embedded into a token stream (consumed by the MMDiT core via
joint attention) plus a pooled summary (added to the modulation vector for
flagged-C decoder modulation).
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike


class ScalarCondEmbedder(nnx.Module):
    """Embed a few condition tokens (e.g. theta scalars) to ``hidden_size``.

    Input ``cond`` is ``(B, k, in_channels)`` (or ``(B, k)``, auto-expanded).
    Returns ``(cond_tokens (B, k, hidden), summary (B, hidden))`` where the
    summary is a projection of the mean-pooled tokens.
    """

    def __init__(self, in_channels, hidden_size, rngs, param_dtype: DTypeLike = jnp.bfloat16):
        self.token_proj = nnx.Linear(
            in_features=in_channels, out_features=hidden_size, use_bias=True,
            rngs=rngs, param_dtype=param_dtype,
        )
        self.summary_proj = nnx.Linear(
            in_features=hidden_size, out_features=hidden_size, use_bias=True,
            rngs=rngs, param_dtype=param_dtype,
        )

    def __call__(self, cond):
        if cond.ndim == 2:
            cond = cond[..., None]
        tokens = self.token_proj(cond)
        summary = self.summary_proj(jnp.mean(tokens, axis=1))
        return tokens, summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_cond.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/cond.py \
        tests/experimental/models/fielddit/test_cond.py
git commit -m "feat(fielddit): ScalarCondEmbedder (tokens + pooled summary)"
```

---

## Task 7: `core.py` — `MMDiTCore`

The transformer bottleneck. Reuses Flux1's `DoubleStreamBlock`/`SingleStreamBlock`/`EmbedND` and the `FeatureEmbedder`. obs tokens carry rope2d ids; the few cond tokens use absolute (order-free) ids (Flux1's mixed-id pattern: add a learned absolute embedding to cond, and give cond dummy zero rope ids so rope is identity on cond positions). The block ordering matches Flux1 exactly: cond is concatenated **before** obs.

**Files:**
- Create: `src/gensbi/experimental/models/fielddit/core.py`
- Test: `tests/experimental/models/fielddit/test_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/experimental/models/fielddit/test_core.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.recipes.utils import init_ids_2d, init_ids_1d
from gensbi.experimental.models.fielddit.core import MMDiTCore


def _make_core():
    # hidden = sum(axes_dim) * num_heads = 8 * 2 = 16; head_dim = 8
    return MMDiTCore(
        hidden_size=16, num_heads=2, mlp_ratio=4.0, depth=1, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=10000, n_cond_tokens=3, qkv_bias=False,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )


def test_mmdit_core_forward_shape_and_finite():
    core = _make_core()
    obs_ids, n_obs = init_ids_2d((8, 8), semantic_id=0, size=2)  # 16 tokens
    cond_ids, _ = init_ids_1d(3)

    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
    vec = jnp.ones((2, 16))

    out = core(obs, cond, vec, obs_ids, cond_ids)
    # returns only the obs tokens, same shape as obs input
    assert out.shape == (2, 16, 16)
    assert jnp.all(jnp.isfinite(out))


def test_mmdit_core_batch1_ids_broadcast():
    """obs_ids/cond_ids have batch dim 1 but obs has batch 4."""
    core = _make_core()
    obs_ids, _ = init_ids_2d((8, 8), semantic_id=0, size=2)
    cond_ids, _ = init_ids_1d(3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 3, 16))
    vec = jnp.ones((4, 16))
    out = core(obs, cond, vec, obs_ids, cond_ids)
    assert out.shape == (4, 16, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_core.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/gensbi/experimental/models/fielddit/core.py`:

```python
"""The MMDiT bottleneck for FieldDiT — Flux1 joint-attention over obs+cond.

obs tokens carry rope2d positional ids; the few cond tokens are absolute
(order-free), embedded with a learned id embedding and given dummy zero rope
ids so the rotary encoding is identity on them. Block order matches Flux1:
cond is concatenated before obs.
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.models.flux1.layers import DoubleStreamBlock, SingleStreamBlock, EmbedND
from gensbi.models.embedding import FeatureEmbedder


class MMDiTCore(nnx.Module):
    """Flux1 double-stream + single-stream transformer over obs+cond tokens.

    Parameters mirror the relevant subset of ``Flux1Params``. ``vec`` (the
    time (+cond summary, +guidance) modulation vector) is supplied externally
    so the same vector can drive the conv codec's AdaGN-zero modulation.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        depth,
        depth_single_blocks,
        axes_dim,
        theta,
        n_cond_tokens,
        qkv_bias,
        rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        assert sum(axes_dim) == head_dim, (
            f"sum(axes_dim)={sum(axes_dim)} must equal head_dim={head_dim}"
        )
        self.pe_embedder = EmbedND(dim=head_dim, theta=theta, axes_dim=list(axes_dim))
        # absolute (order-free) id embedding for the few cond tokens
        self.cond_ids_embedder = FeatureEmbedder(
            num_embeddings=n_cond_tokens,
            hidden_size=hidden_size,
            kind="absolute",
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.double_blocks = nnx.Sequential(
            *[
                DoubleStreamBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qkv_features=hidden_size, qkv_bias=qkv_bias,
                    rngs=rngs, param_dtype=param_dtype,
                )
                for _ in range(depth)
            ]
        )
        self.single_blocks = nnx.Sequential(
            *[
                SingleStreamBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qkv_features=hidden_size, rngs=rngs, param_dtype=param_dtype,
                )
                for _ in range(depth_single_blocks)
            ]
        )

    def __call__(self, obs_tokens, cond_tokens, vec, obs_ids, cond_ids):
        B = obs_tokens.shape[0]
        if obs_ids.shape[0] == 1 and B > 1:
            obs_ids = jnp.repeat(obs_ids, B, axis=0)

        # absolute id embedding added to the cond value embedding (Flux1 pattern)
        cond_tokens = cond_tokens * jnp.sqrt(self.hidden_size) + self.cond_ids_embedder(cond_ids)

        # dummy zero rope ids for cond so rope is identity on cond positions
        cond_ids_rope = jnp.zeros(
            (obs_ids.shape[0], cond_tokens.shape[1], obs_ids.shape[2]), dtype=obs_ids.dtype
        )
        ids = jnp.concatenate((cond_ids_rope, obs_ids), axis=1)
        pe = self.pe_embedder(ids)

        for blk in self.double_blocks.layers:
            obs_tokens, cond_tokens = blk(obs=obs_tokens, cond=cond_tokens, vec=vec, pe=pe)

        x = jnp.concatenate((cond_tokens, obs_tokens), axis=1)
        for blk in self.single_blocks.layers:
            x = blk(x, vec=vec, pe=pe)

        return x[:, cond_tokens.shape[1]:, ...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_core.py -v`
Expected: PASS (2 tests).

> **Note for the implementer:** if `apply_rope` complains about a batch-dim mismatch between `pe` and the attention tensors, the `jnp.repeat(obs_ids, B, axis=0)` guard already makes `pe` batch `B`; if a problem persists, confirm `pe`'s leading dim broadcasts against `q` of shape `(B, H, L, D)`. This is a small detail to resolve at green-test time, not a design change.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/core.py \
        tests/experimental/models/fielddit/test_core.py
git commit -m "feat(fielddit): MMDiTCore (Flux1 joint attention, rope2d obs + absolute cond)"
```

---

## Task 8: `model.py` — `FieldDiTParams`

The config dataclass (mirrors `Flux1Params`). Derives `hidden_size` from `axes_dim × num_heads`, the downsample depth, the meeting-grid feature shape, the token grid, the obs token count, and precomputes the obs/cond ids. Validates divisibility and rope constraints.

**Files:**
- Create: `src/gensbi/experimental/models/fielddit/model.py` (params only this task)
- Test: `tests/experimental/models/fielddit/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/experimental/models/fielddit/test_model.py`:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.experimental.models.fielddit.model import FieldDiTParams


def _params(**overrides):
    base = dict(
        in_channels=1,
        field_shape=(32, 32),
        encoder_widths=(8, 16, 32),  # D = 2
        cond_dim=3,
        rngs=nnx.Rngs(0),
        num_heads=2,
        axes_dim=[2, 2, 4],          # sum 8 -> hidden 16
        patch_size=2,
        param_dtype=jnp.float32,
    )
    base.update(overrides)
    return FieldDiTParams(**base)


def test_params_derive_hidden_and_grid():
    p = _params()
    assert p.hidden_size == 16           # sum([2,2,4]) * num_heads(2)
    assert p.depth_levels == 2           # len(encoder_widths) - 1
    assert (p.feat_h, p.feat_w) == (8, 8)  # 32 / 2**2
    assert p.token_grid == (4, 4)        # feat / patch_size
    assert p.n_obs_tokens == 16
    assert p.obs_ids.shape == (1, 16, 3)
    assert p.cond_ids.shape == (1, 3, 1)


def test_params_reject_indivisible_field_shape():
    with pytest.raises(AssertionError, match="divisible"):
        _params(field_shape=(30, 32))  # 30 not divisible by 2**2


def test_params_reject_odd_axes_dim():
    with pytest.raises(AssertionError, match="even"):
        _params(axes_dim=[3, 2, 3])  # odd entries invalid for rope


def test_params_reject_wrong_axes_len():
    with pytest.raises(AssertionError, match="3 entries"):
        _params(axes_dim=[4, 4])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py -v`
Expected: FAIL with `ImportError` (no `model` module).

- [ ] **Step 3: Write minimal implementation**

Create `src/gensbi/experimental/models/fielddit/model.py` (params block; the `FieldDiT` class is appended in Task 9):

```python
"""FieldDiT config and assembly.

FieldDiT = conv U-Net (ObsEncoder/ObsDecoder) with an MMDiT transformer
bottleneck, for conditional pixel-space flow matching on 2D fields.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.recipes.utils import init_ids_1d, init_ids_2d
from gensbi.models.flux1.layers import MLPEmbedder, timestep_embedding, Identity

from gensbi.experimental.models.fielddit.codec import (
    ObsEncoder,
    ObsDecoder,
    Tokenizer,
    Untokenizer,
)
from gensbi.experimental.models.fielddit.cond import ScalarCondEmbedder
from gensbi.experimental.models.fielddit.core import MMDiTCore


@dataclass
class FieldDiTParams:
    """Configuration for :class:`FieldDiT` (mirrors the style of ``Flux1Params``).

    The meeting-grid token count is **derived** from ``field_shape``,
    ``encoder_widths`` (depth) and ``patch_size`` — it is not prescribed:
    ``tokens = (H / (2**D * p)) * (W / (2**D * p))`` with ``D = len(encoder_widths) - 1``.

    Parameters
    ----------
    in_channels : int
        Channels of the field (and of the velocity output).
    field_shape : tuple of int
        ``(H, W)`` spatial shape of the field. Must be divisible by ``2**D``
        and the resulting feature grid divisible by ``patch_size``.
    encoder_widths : tuple of int
        Channel width per resolution; length ``D + 1`` (last = bottleneck).
    cond_dim : int
        Number of conditioning tokens (e.g. number of theta scalars).
    rngs : nnx.Rngs
        Random number generators.
    cond_in_channels : int
        Features per conditioning token. Default 1.
    res_blocks_down, res_blocks_up : int
        Residual blocks per encoder/decoder stage. Default 2.
    patch_size : int
        Patch edge length at the meeting grid. Default 2.
    num_heads : int
        Attention heads. Default 12.
    axes_dim : list of int, optional
        RoPE dims for the 3 obs id axes (semantic, h, w); each even, summing to
        ``hidden_size // num_heads``. Defaults to ``[16, 24, 24]``.
    mlp_ratio : float
        Transformer MLP ratio. Default 4.0.
    depth, depth_single_blocks : int
        Number of double-stream / single-stream blocks. Default 2 each.
    qkv_bias : bool
        Bias in attention QKV. Default False.
    theta : int
        RoPE base. Default 10000.
    use_cond_summary_in_vec : bool
        flagged-C: add the pooled condition summary to the modulation vector
        (drives decoder + core modulation). Default True.
    norm_groups : int
        Target GroupNorm groups for conv blocks (reduced per-width via gcd).
        Default 8.
    guidance_embed : bool
        Enable an optional guidance modulation hook. Default False.
    vec_in_dim : int, optional
        Input dim for the guidance MLP (required iff ``guidance_embed``).
    param_dtype : DTypeLike
        Parameter dtype. Default ``jnp.bfloat16``.
    """

    in_channels: int
    field_shape: Tuple[int, int]
    encoder_widths: Tuple[int, ...]
    cond_dim: int
    rngs: nnx.Rngs
    cond_in_channels: int = 1
    res_blocks_down: int = 2
    res_blocks_up: int = 2
    patch_size: int = 2
    num_heads: int = 12
    axes_dim: Optional[List[int]] = None
    mlp_ratio: float = 4.0
    depth: int = 2
    depth_single_blocks: int = 2
    qkv_bias: bool = False
    theta: int = 10000
    use_cond_summary_in_vec: bool = True
    norm_groups: int = 8
    guidance_embed: bool = False
    vec_in_dim: Optional[int] = None
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        if self.axes_dim is None:
            self.axes_dim = [16, 24, 24]
        assert len(self.axes_dim) == 3, "axes_dim must have 3 entries (semantic, h, w)"
        for a in self.axes_dim:
            assert a % 2 == 0, f"each axes_dim entry must be even for rope, got {self.axes_dim}"

        self.hidden_size = int(sum(self.axes_dim) * self.num_heads)
        self.depth_levels = len(self.encoder_widths) - 1

        H, W = self.field_shape
        factor = 2 ** self.depth_levels
        assert H % factor == 0 and W % factor == 0, (
            f"field_shape {self.field_shape} must be divisible by 2**D={factor}"
        )
        self.feat_h = H // factor
        self.feat_w = W // factor

        p = self.patch_size
        assert self.feat_h % p == 0 and self.feat_w % p == 0, (
            f"meeting grid ({self.feat_h},{self.feat_w}) must be divisible by patch_size {p}"
        )
        self.token_grid = (self.feat_h // p, self.feat_w // p)
        self.n_obs_tokens = self.token_grid[0] * self.token_grid[1]

        self.obs_ids, _ = init_ids_2d((self.feat_h, self.feat_w), semantic_id=0, size=p)
        self.cond_ids, _ = init_ids_1d(self.cond_dim, semantic_id=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py \
        tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): FieldDiTParams config with derived meeting grid"
```

---

## Task 9: `model.py` — `FieldDiT` assembly + forward

Wire the components into the end-to-end network. Forward contract: `model(t, obs=field, cond, *, guidance=None) -> velocity field` (same shape as `obs`); the meeting-grid ids are built internally. `obs_ids`/`cond_ids`/`conditioned` kwargs are accepted but ignored (drop-in compatibility with `ConditionalWrapper`'s call convention). Encoder is modulated by **time only**; decoder + core by `time (+ summary, + guidance)`.

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/model.py` (append `FieldDiT`)
- Test: `tests/experimental/models/fielddit/test_model.py` (append)

- [ ] **Step 1: Write the failing test (append to `test_model.py`)**

```python
from gensbi.experimental.models.fielddit.model import FieldDiT


def _small_model(seed=0):
    return FieldDiT(_params(rngs=nnx.Rngs(seed)))


def test_fielddit_forward_shape_and_zero_init():
    model = _small_model()
    B = 2
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (B, 3, 1))
    t = jnp.ones((B,))
    v = model(t, obs, cond)
    assert v.shape == obs.shape
    # zero-init decoder conv_out => exactly-zero velocity at init (also proves
    # no NaN anywhere upstream, since 0 * NaN == NaN).
    assert jnp.allclose(v, 0.0)


def test_fielddit_handles_batch_sizes():
    model = _small_model()
    for B in (1, 4):
        obs = jax.random.normal(jax.random.PRNGKey(B), (B, 32, 32, 1))
        cond = jax.random.normal(jax.random.PRNGKey(B + 100), (B, 3, 1))
        t = jnp.ones((B,))
        v = model(t, obs, cond)
        assert v.shape == (B, 32, 32, 1)


def test_fielddit_ignores_extra_kwargs():
    """Accepts (and ignores) obs_ids/cond_ids/conditioned for wrapper compat."""
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))
    v = model(t, obs, cond, obs_ids="ignored", cond_ids="ignored", conditioned=True)
    assert v.shape == obs.shape


def test_fielddit_is_differentiable():
    model = _small_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))
    t = jnp.ones((2,))

    # NOTE: use a non-zero target so the loss does not vanish at v == 0. With
    # mean(v**2) the gradient is identically zero at init (v == 0), and the
    # zero-init output conv blocks gradient to pre-final params at step 0 — both
    # are expected for a zero-init output layer, so we only assert finiteness +
    # a non-zero gradient on the final conv (the output path is connected).
    def loss_fn(model):
        return jnp.mean((model(t, obs, cond) - 1.0) ** 2)

    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    conv_out_grad = nnx.state(grads, nnx.Param)["decoder"]["conv_out"]["kernel"].value
    assert jnp.any(jnp.abs(conv_out_grad) > 0)


def test_fielddit_param_dtype_propagates():
    params = _params(rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    model = FieldDiT(params)
    assert model.time_in.in_layer.kernel[...].dtype == jnp.bfloat16
    assert model.decoder.conv_out.kernel[...].dtype == jnp.bfloat16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py -v`
Expected: FAIL with `ImportError` (`FieldDiT` not defined).

- [ ] **Step 3: Write minimal implementation (append to `model.py`)**

```python
class FieldDiT(nnx.Module):
    """Conditional flow-matching network for 2D fields (pixel space).

    Forward: ``(t, obs=field, cond) -> velocity field`` of the same shape. The
    conv encoder is modulated by time only; the decoder and the MMDiT core are
    modulated by ``vec = time (+ cond summary if flagged-C) (+ guidance)``. The
    meeting-grid rope2d obs ids and absolute cond ids are built internally.
    """

    def __init__(self, params: FieldDiTParams):
        self.params = params
        p = params
        hid = p.hidden_size

        self.encoder = ObsEncoder(
            p.in_channels, p.encoder_widths, p.res_blocks_down,
            vec_dim=hid, norm_groups=p.norm_groups, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        c_bottleneck = p.encoder_widths[-1]
        self.tokenizer = Tokenizer(
            c_bottleneck, p.patch_size, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.cond_embedder = ScalarCondEmbedder(
            p.cond_in_channels, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.core = MMDiTCore(
            hid, p.num_heads, p.mlp_ratio, p.depth, p.depth_single_blocks,
            axes_dim=p.axes_dim, theta=p.theta, n_cond_tokens=p.cond_dim,
            qkv_bias=p.qkv_bias, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        self.untokenizer = Untokenizer(
            c_bottleneck, p.patch_size, hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        self.decoder = ObsDecoder(
            p.in_channels, p.encoder_widths, p.res_blocks_up,
            vec_dim=hid, norm_groups=p.norm_groups, rngs=p.rngs, param_dtype=p.param_dtype,
        )
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=hid, rngs=p.rngs, param_dtype=p.param_dtype
        )
        if p.guidance_embed:
            assert p.vec_in_dim is not None, "vec_in_dim required when guidance_embed=True"
            self.guidance_in = MLPEmbedder(
                p.vec_in_dim, hid, rngs=p.rngs, param_dtype=p.param_dtype
            )
        else:
            self.guidance_in = Identity()

        # precomputed ids (stored as plain arrays; broadcast over batch in core)
        self.obs_ids = p.obs_ids
        self.cond_ids = p.cond_ids
        self.token_grid = p.token_grid

    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,      # accepted & ignored (ids built internally)
        cond_ids=None,     # accepted & ignored
        conditioned=True,  # accepted & ignored
        guidance=None,
    ):
        p = self.params
        obs = jnp.asarray(obs, dtype=p.param_dtype)
        cond = jnp.asarray(cond, dtype=p.param_dtype)
        t = jnp.asarray(t, dtype=p.param_dtype)

        time_vec = self.time_in(timestep_embedding(t, 256))  # (B, hidden)
        cond_tokens, summary = self.cond_embedder(cond)       # (B, k, hidden), (B, hidden)

        vec = time_vec
        if p.use_cond_summary_in_vec:
            vec = vec + summary
        if p.guidance_embed:
            if guidance is None:
                raise ValueError("guidance required when guidance_embed=True")
            vec = vec + self.guidance_in(guidance)

        feat, pos_skips, neg_skips = self.encoder(obs, time_vec)   # time-only modulation
        obs_tokens = self.tokenizer(feat)
        obs_tokens = self.core(obs_tokens, cond_tokens, vec, self.obs_ids, self.cond_ids)
        feat = self.untokenizer(obs_tokens, self.token_grid)
        v = self.decoder(feat, vec, pos_skips, neg_skips)         # time + cond modulation
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py -v`
Expected: PASS (all model tests).

- [ ] **Step 5: Run the full FieldDiT test suite**

Run: `uv run pytest tests/experimental/models/fielddit/ -v`
Expected: PASS (all tests across `test_blocks`, `test_codec`, `test_cond`, `test_core`, `test_model`).

- [ ] **Step 6: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/model.py \
        tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): FieldDiT assembly and end-to-end forward"
```

---

## Task 10: Package exports

Expose `FieldDiT`/`FieldDiTParams` from the package and the experimental models namespace.

**Files:**
- Modify: `src/gensbi/experimental/models/fielddit/__init__.py`
- Modify: `src/gensbi/experimental/models/__init__.py:1-11`
- Test: `tests/experimental/models/fielddit/test_model.py` (append an import test)

- [ ] **Step 1: Write the failing test (append to `test_model.py`)**

```python
def test_public_exports():
    from gensbi.experimental.models.fielddit import FieldDiT as FD, FieldDiTParams as FDP
    from gensbi.experimental.models import FieldDiT as FD2, FieldDiTParams as FDP2

    assert FD is FD2
    assert FDP is FDP2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py::test_public_exports -v`
Expected: FAIL with `ImportError` (names not exported yet).

- [ ] **Step 3: Update the package `__init__.py`**

Replace the contents of `src/gensbi/experimental/models/fielddit/__init__.py` with:

```python
"""FieldDiT: conditional flow-matching for 2D field-level inference (Phase 1)."""

from gensbi.experimental.models.fielddit.model import FieldDiT, FieldDiTParams

__all__ = [
    "FieldDiT",
    "FieldDiTParams",
]
```

- [ ] **Step 4: Update the experimental models `__init__.py`**

Edit `src/gensbi/experimental/models/__init__.py` so it reads:

```python
from .autoencoders import AutoEncoder1D, AutoEncoder2D, AutoEncoderParams, vae_loss_fn
from .glue import Embedded1DModel, Embedded2DModel
from .fielddit import FieldDiT, FieldDiTParams

__all__ = [
    "AutoEncoder1D",
    "AutoEncoder2D",
    "AutoEncoderParams",
    "vae_loss_fn",
    "Embedded1DModel",
    "Embedded2DModel",
    "FieldDiT",
    "FieldDiTParams",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/experimental/models/fielddit/test_model.py::test_public_exports -v`
Expected: PASS.

- [ ] **Step 6: Run the full experimental models suite (no regressions)**

Run: `uv run pytest tests/experimental/models/ -v`
Expected: PASS (FieldDiT tests + existing autoencoders/glue tests unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/gensbi/experimental/models/fielddit/__init__.py \
        src/gensbi/experimental/models/__init__.py \
        tests/experimental/models/fielddit/test_model.py
git commit -m "feat(fielddit): export FieldDiT and FieldDiTParams"
```

---

## Done criteria

- `uv run pytest tests/experimental/models/fielddit/ -v` is green.
- `uv run pytest tests/experimental/models/ -v` is green (no regressions).
- `from gensbi.experimental.models import FieldDiT, FieldDiTParams` works.
- A `FieldDiT` built from `FieldDiTParams` maps `(t, field, cond) -> velocity field` of the same shape, is finite, is exactly zero at init (zero-init output conv), and is differentiable.

## Notes for the next plan (not built here)

- **Pipeline wiring:** `FMLoss` calls `model(obs=x_t, t=t, **model_extras)` and compares the output to `path_sample.dx_t` elementwise — `FieldDiT`'s field-shaped I/O is already compatible, but `ConditionalPipeline` flattens `dim_obs` to a token count and sets `event_shape=(dim_obs, ch_obs)`, which is wrong for pixel-space fields. A thin field pipeline (override `event_shape = (H, W, C)`, skip id-resolution, pass `cond` raw) is the clean follow-on.
- **`ConditionalWrapper` compatibility:** `_expand_dims` only acts when `ndim < 3`, so a 4D field passes through untouched; the wrapper may be reusable as-is.
- **GRF 256² validation** (power-spectrum recovery, field-space SBC/TARP) is the separate experimental plan referenced by spec §6. Note there is no existing `grf.py` — that validation is greenfield.
- **Open spec flag (§2, §9 Q1):** GroupNorm's spatial pooling may harm emulation of spatial-statistic targets; worth an ablation against the normalization choice before scaling up.
