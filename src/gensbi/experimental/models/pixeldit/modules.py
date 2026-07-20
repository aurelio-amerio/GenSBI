"""Low-level building blocks for the PixelDiT port.

Faithful port of ``reference/PixelDiT/pixdit_core/modules.py`` into flax.nnx.
Deliberate deviations from the reference are called out inline.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import DTypeLike


# ---------------------------------------------------------------------------
# Buffer — non-trainable array storage (rope/sincos tables)
# ---------------------------------------------------------------------------


class Buffer(nnx.Variable):
    """Non-trainable array buffer (rope tables, sincos tables, etc.).

    A dedicated Variable type so buffers are:
    - filterable with ``nnx.state(model, Buffer)``,
    - excluded from ``nnx.Param`` state (safe from dtype casts applied to
      parameter state), and
    - importable by Tasks 3 and 6 which need the same distinction.

    Mirrors ``RopeIds`` in ``fielddit/model.py:28-34``.
    """


# ---------------------------------------------------------------------------
# SwiGLU (ref FeedForward, modules.py:119-129)
# ---------------------------------------------------------------------------


class SwiGLU(nnx.Module):
    """SwiGLU feed-forward (ref ``FeedForward``, modules.py:119-129).

    ``hidden = int(2 * (dim * mlp_ratio) / 3)``; all three projections are
    bias-free.  Forward: ``w2(silu(w1(x)) * w3(x))``.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        # int(2 * (dim * mlp_ratio) / 3) is identical to ref's int(2 * int(dim*mlp_ratio) / 3)
        # for integer dim*mlp_ratio; single-int form avoids the redundant inner int().
        hidden = int(2 * (dim * mlp_ratio) / 3)
        self.w1 = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.w3 = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.w2 = nnx.Linear(hidden, dim, use_bias=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x):
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# PixelMLP (ref MLP, modules.py:223-238)
# ---------------------------------------------------------------------------


class PixelMLP(nnx.Module):
    """Standard GELU MLP used in pixel-level DiT blocks (ref ``MLP``, modules.py:223-238).

    ``Linear(dim → 4·dim)`` → GELU → ``Linear(→ dim)``.  Both linears have
    biases; no dropout (we never train with dropout).
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.fc2 = nnx.Linear(hidden, dim, use_bias=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)  # exact erf-GELU; ref uses nn.GELU() (PyTorch default = erf)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# TimestepConditioner (ref modules.py:63-91)
# ---------------------------------------------------------------------------


def _timestep_embedding(t, dim: int, max_period: float = 10.0):
    """Sinusoidal timestep embedding in float32 (ref ``TimestepConditioner.timestep_embedding``).

    Deliberate deviations from ``flux1.timestep_embedding``:
    - ``max_period = 10`` (not 10000).
    - No ×1000 time factor.
    - Order is ``cat([cos, sin])`` (reference line 80).
    - Always computed in float32 (FieldDiT lesson: bf16 ``t`` quantizes fine
      differences; cast to param_dtype only before the MLP).
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(half, dtype=jnp.float32)
        / half
    )
    # t is kept in float32 regardless of its input dtype
    args = t.astype(jnp.float32)[..., None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[..., :1])], axis=-1
        )
    return embedding  # float32


class TimestepConditioner(nnx.Module):
    """Embed a scalar timestep into ``hidden_size`` (ref modules.py:63-91).

    The sinusoid is always computed in float32; ``mlp_in`` (a compute-dtype
    Linear) downcasts it to the compute dtype via its own ``promote_dtype``,
    so no manual cast is needed at the call site.
    MLP weights are initialised with ``normal(std=0.02)`` (ref
    ``initialize_weights``).
    """

    def __init__(
        self,
        hidden_size: int,
        freq_dim: int = 256,
        *,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.freq_dim = freq_dim
        init = jax.nn.initializers.normal(stddev=0.02)
        self.mlp_in = nnx.Linear(
            freq_dim,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
        )
        self.mlp_out = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
        )

    def __call__(self, t):
        # Sinusoid always computed in float32 (fp32 island); mlp_in's own
        # promote_dtype downcasts it to the compute dtype for the matmul.
        t_freq = _timestep_embedding(t, self.freq_dim)  # float32
        t_emb = nnx.silu(self.mlp_in(t_freq))
        t_emb = self.mlp_out(t_emb)
        return t_emb


# ---------------------------------------------------------------------------
# FinalLayer (ref modules.py:241-250)
# ---------------------------------------------------------------------------


class FinalLayer(nnx.Module):
    """RMSNorm + zero-init linear output projection (ref modules.py:241-250).

    The linear's kernel and bias are both zero-initialized, so the layer is
    exactly the zero map at initialisation (safe residual scaling).

    models-emit-fp32 contract: both the norm and the linear are constructed
    with ``dtype=jnp.float32`` (never a post-hoc ``.astype``), regardless of
    the compute-dtype knob used elsewhere in the model -- this is the
    designated emit-fp32 endpoint, so it has no separate ``dtype`` parameter.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=1e-6,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=param_dtype,
        )
        self.linear = nnx.Linear(
            hidden_size,
            out_channels,
            use_bias=True,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# 2D sincos positional embedding (ref modules.py:10-56)
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim: int, h: int, w: int) -> np.ndarray:
    """Pure-numpy 2D sincos positional embedding table.

    Returns ``(h*w, embed_dim)`` float32.  Supports ``h != w``.

    The axis convention is copied exactly from the reference (modules.py:16-22):
    ``grid = meshgrid(grid_w, grid_h)`` — w goes first — so ``grid[0]`` is the
    w-axis and ``grid[1]`` is the h-axis.  ``emb_h`` is built from ``grid[0]``
    (the w-grid) following the reference naming; do not "fix" this.
    """
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    # Reference line 18: "here w goes first"
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, h, w)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb.astype(np.float32)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sincos embedding for a grid of positions (ref modules.py:38-56).

    Returns ``(M, embed_dim)`` where the order is ``[sin, cos]`` (reference
    line 55: ``cat([emb_sin, emb_cos])``).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
