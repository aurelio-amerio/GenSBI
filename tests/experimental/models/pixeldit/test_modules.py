"""Tests for PixelDiT low-level modules.

TDD: failing first, then implemented.
"""

import math

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import pytest
from flax import nnx

from gensbi.experimental.models.pixeldit.modules import (
    Buffer,
    FinalLayer,
    PixelMLP,
    SwiGLU,
    TimestepConditioner,
    _timestep_embedding,
    get_2d_sincos_pos_embed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=seed)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


def test_buffer_is_not_param():
    """Buffer must be a distinct Variable so it's excluded from nnx.Param state."""
    assert issubclass(Buffer, nnx.Variable)
    assert Buffer is not nnx.Param


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def test_swiglu_hidden_width():
    """hidden = int(2 * (dim * mlp_ratio) / 3)."""
    dim, mlp_ratio = 64, 4.0
    expected_hidden = int(2 * (dim * mlp_ratio) / 3)  # 170
    m = SwiGLU(dim, mlp_ratio=mlp_ratio, rngs=make_rngs())
    assert m.w1.in_features == dim
    assert m.w1.out_features == expected_hidden
    assert m.w2.in_features == expected_hidden
    assert m.w2.out_features == dim
    assert m.w3.in_features == dim
    assert m.w3.out_features == expected_hidden


def test_swiglu_no_bias():
    m = SwiGLU(32, rngs=make_rngs())
    assert m.w1.use_bias is False
    assert m.w2.use_bias is False
    assert m.w3.use_bias is False


def test_swiglu_output_shape():
    dim = 48
    m = SwiGLU(dim, rngs=make_rngs())
    x = jnp.ones((2, 5, dim))
    y = m(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# PixelMLP
# ---------------------------------------------------------------------------


def test_pixelmlp_hidden_width():
    dim, mlp_ratio = 32, 4.0
    m = PixelMLP(dim, mlp_ratio=mlp_ratio, rngs=make_rngs())
    expected_hidden = int(dim * mlp_ratio)
    assert m.fc1.out_features == expected_hidden
    assert m.fc2.out_features == dim


def test_pixelmlp_has_bias():
    m = PixelMLP(32, rngs=make_rngs())
    assert m.fc1.use_bias is True
    assert m.fc2.use_bias is True


def test_pixelmlp_output_shape():
    dim = 32
    m = PixelMLP(dim, rngs=make_rngs())
    x = jnp.ones((2, 7, dim))
    y = m(x)
    assert y.shape == x.shape


def test_pixelmlp_uses_erf_gelu():
    """PixelMLP must use exact erf-GELU, not tanh-approximate.

    Strategy: build a PixelMLP in float32 with fc1 acting as identity on the
    first dim channels (top-left block = I, rest zero) and fc2 also identity,
    biases zero.  Then output[:, :dim] ≈ gelu(x[:, :dim]).  We probe x=2.0
    where erf-GELU and tanh-GELU differ by >1e-4.
    """
    dim = 8
    m = PixelMLP(dim, mlp_ratio=4.0, rngs=make_rngs(), param_dtype=jnp.float32)
    hidden = int(dim * 4.0)  # 32

    # Set fc1: (dim → hidden) kernel.  Top dim×dim block = I, rest = 0; bias = 0.
    k1 = np.zeros((dim, hidden), dtype=np.float32)
    k1[:dim, :dim] = np.eye(dim)
    m.fc1.kernel.set_value(jnp.array(k1))
    m.fc1.bias.set_value(jnp.zeros(hidden, dtype=jnp.float32))

    # Set fc2: (hidden → dim) kernel.  Left dim×dim block = I; bias = 0.
    k2 = np.zeros((hidden, dim), dtype=np.float32)
    k2[:dim, :dim] = np.eye(dim)
    m.fc2.kernel.set_value(jnp.array(k2))
    m.fc2.bias.set_value(jnp.zeros(dim, dtype=jnp.float32))

    # x=1.5: erf-GELU ≈ 1.39979, tanh-GELU ≈ 1.39957, diff ≈ 2.2e-4 (well above 1e-4 threshold)
    probe = 1.5
    x = jnp.full((1, 1, dim), probe, dtype=jnp.float32)
    y = m(x)  # shape (1, 1, dim)

    # erf-GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    erf_gelu = 0.5 * probe * (1.0 + float(jax.scipy.special.erf(probe / math.sqrt(2))))
    # tanh-GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    tanh_gelu = float(jax.nn.gelu(jnp.array(probe), approximate=True))

    got = float(y[0, 0, 0])
    np.testing.assert_allclose(got, erf_gelu, atol=1e-5,
                               err_msg="PixelMLP GELU must be exact erf variant")
    # Confirm the two variants actually differ enough to distinguish
    assert abs(erf_gelu - tanh_gelu) > 1e-4, "probe value does not distinguish GELU variants"
    assert abs(got - tanh_gelu) > 1e-4, "PixelMLP appears to use tanh-approximate GELU"


# ---------------------------------------------------------------------------
# TimestepConditioner
# ---------------------------------------------------------------------------


def test_timestep_conditioner_output_shape():
    hidden = 64
    m = TimestepConditioner(hidden, freq_dim=256, rngs=make_rngs())
    t = jnp.linspace(0.0, 1.0, 8)
    out = m(t)
    assert out.shape == (8, hidden)


def test_timestep_conditioner_finite():
    m = TimestepConditioner(64, freq_dim=256, rngs=make_rngs(), param_dtype=jnp.bfloat16)
    t = jnp.linspace(0.0, 1.0, 8)
    out = m(t)
    assert jnp.all(jnp.isfinite(out))


def test_timestep_conditioner_resolves_small_t():
    """max_period=10 sinusoid in float32 must distinguish t=0.0 from t=0.001."""
    m = TimestepConditioner(64, freq_dim=256, rngs=make_rngs(), param_dtype=jnp.bfloat16)
    t0 = jnp.array([0.0])
    t1 = jnp.array([0.001])
    out0 = m(t0)
    out1 = m(t1)
    # If computed in bf16, tiny t differences get quantized to zero; f32 keeps them.
    assert not jnp.allclose(out0, out1, atol=1e-4), (
        "TimestepConditioner should distinguish t=0.0 from t=0.001 "
        "(sinusoid must be computed in float32)"
    )


def test_timestep_conditioner_param_dtype():
    """MLP weights should be stored in param_dtype."""
    m = TimestepConditioner(64, freq_dim=256, rngs=make_rngs(), param_dtype=jnp.bfloat16)
    # Check that both linear layers' kernels are bfloat16
    assert m.mlp_in.kernel.get_value().dtype == jnp.bfloat16
    assert m.mlp_out.kernel.get_value().dtype == jnp.bfloat16


def test_timestep_embedding_ordering_at_t0():
    """At t=0: first freq_dim//2 features are cos(0)=1, last half are sin(0)=0.

    Pins the cat([cos, sin]) order and max_period from _timestep_embedding.
    """
    freq_dim = 256
    t = jnp.array([0.0])
    emb = _timestep_embedding(t, freq_dim)  # (1, freq_dim)
    assert emb.shape == (1, freq_dim)

    half = freq_dim // 2
    cos_part = emb[0, :half]
    sin_part = emb[0, half:]

    # cos(0 * freqs) = cos(0) = 1 for all frequencies
    np.testing.assert_allclose(np.asarray(cos_part), 1.0, atol=1e-6,
                               err_msg="First half of t=0 embedding must be all-ones (cos)")
    # sin(0 * freqs) = sin(0) = 0 for all frequencies
    np.testing.assert_allclose(np.asarray(sin_part), 0.0, atol=1e-6,
                               err_msg="Last half of t=0 embedding must be all-zeros (sin)")


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------


def test_finallayer_output_shape():
    hidden, out_ch = 64, 16
    m = FinalLayer(hidden, out_ch, rngs=make_rngs())
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, hidden))
    y = m(x)
    assert y.shape == (2, 10, out_ch)


def test_finallayer_zero_output_at_init():
    """Linear kernel and bias are zero-initialized; output must be exactly zero."""
    hidden, out_ch = 64, 8
    m = FinalLayer(hidden, out_ch, rngs=make_rngs())
    x = jax.random.normal(jax.random.PRNGKey(42), (3, 5, hidden))
    y = m(x)
    np.testing.assert_array_equal(np.asarray(y), 0.0)


# ---------------------------------------------------------------------------
# get_2d_sincos_pos_embed
# ---------------------------------------------------------------------------


def _reference_sincos(embed_dim, h, w):
    """Inline reference: copy of reference modules.py:10-55 logic for non-square."""
    def get_1d(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000 ** omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first (axis convention kept)
    grid = np.stack(grid, axis=0).reshape(2, 1, h, w)
    emb_h = get_1d(embed_dim // 2, grid[0])
    emb_w = get_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


def test_sincos_shape_square():
    out = get_2d_sincos_pos_embed(64, 8, 8)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32


def test_sincos_shape_nonsquare():
    h, w = 4, 6
    out = get_2d_sincos_pos_embed(64, h, w)
    assert out.shape == (h * w, 64)
    assert out.dtype == np.float32


def test_sincos_matches_reference_square():
    embed_dim, h, w = 64, 8, 8
    ref = _reference_sincos(embed_dim, h, w)
    got = get_2d_sincos_pos_embed(embed_dim, h, w)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_sincos_matches_reference_nonsquare():
    embed_dim, h, w = 64, 4, 6
    ref = _reference_sincos(embed_dim, h, w)
    got = get_2d_sincos_pos_embed(embed_dim, h, w)
    np.testing.assert_allclose(got, ref, atol=1e-6)
