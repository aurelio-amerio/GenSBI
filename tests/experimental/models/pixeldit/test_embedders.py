"""Tests for PixelDiT token embedders.

TDD: failing first, then implemented.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from gensbi.experimental.models.pixeldit.embedders import (
    CondTokenEmbedder,
    PatchTokenEmbedder,
    PixelTokenEmbedder,
    patchify,
    unpatchify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=seed)


# ---------------------------------------------------------------------------
# patchify / unpatchify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [2, 4])
def test_patchify_unpatchify_roundtrip(p):
    """patchify then unpatchify recovers the original tensor exactly."""
    B, H, W, C = 2, 8, 12, 3
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (B, H, W, C))

    tokens = patchify(x, p)
    Hs, Ws = H // p, W // p
    assert tokens.shape == (B, Hs * Ws, p * p * C)

    grid = (Hs, Ws)
    x_rec = unpatchify(tokens, grid, p, C)
    assert x_rec.shape == (B, H, W, C)
    np.testing.assert_array_equal(
        np.asarray(x_rec), np.asarray(x),
        err_msg="patchify → unpatchify must be bitwise exact (pure reshape/transpose)",
    )


def test_patchify_pixel_indexing():
    """Pixel (i, j) of patch (a, b) in the grouped tensor must equal input pixel (a*p+i, b*p+j).

    This pins the (Hs, p, Ws, p) reshape + transpose ordering.
    """
    p = 2
    B, H, W, C = 1, 4, 6, 1  # Hs=2, Ws=3
    # Fill with unique values so every element can be individually identified.
    x_np = np.arange(B * H * W * C, dtype=np.float32).reshape(B, H, W, C)
    x = jnp.array(x_np)

    tokens = patchify(x, p)  # (B, L, p²·C); L = Hs*Ws
    Hs, Ws = H // p, W // p

    # tokens[b, a*Ws+b_col, i*p+j] should equal x[b, a*p+i, b_col*p+j, 0]
    for a in range(Hs):
        for b_col in range(Ws):
            patch_idx = a * Ws + b_col
            for i in range(p):
                for j in range(p):
                    pixel_in_patch = i * p + j
                    got = float(tokens[0, patch_idx, pixel_in_patch])
                    expected = float(x_np[0, a * p + i, b_col * p + j, 0])
                    assert got == expected, (
                        f"tokens[0,{patch_idx},{pixel_in_patch}]={got} "
                        f"!= x[0,{a*p+i},{b_col*p+j},0]={expected}"
                    )


# ---------------------------------------------------------------------------
# PatchTokenEmbedder
# ---------------------------------------------------------------------------


def test_patch_token_embedder_output_shape():
    in_features, hidden_size = 12, 64
    m = PatchTokenEmbedder(in_features, hidden_size, rngs=make_rngs())
    x = jnp.ones((2, 10, in_features))
    y = m(x)
    assert y.shape == (2, 10, hidden_size)


def test_patch_token_embedder_has_bias():
    m = PatchTokenEmbedder(8, 32, rngs=make_rngs())
    assert m.proj.use_bias is True


def test_patch_token_embedder_xavier_kernel():
    """Kernel must be initialized xavier_uniform (non-zero, bounded)."""
    m = PatchTokenEmbedder(16, 32, rngs=make_rngs(), param_dtype=jnp.float32)
    w = np.asarray(m.proj.kernel.get_value())
    assert not np.allclose(w, 0.0), "xavier_uniform kernel must be non-zero"
    # Xavier uniform bound: sqrt(6/(fan_in+fan_out)) = sqrt(6/48) ≈ 0.354
    limit = np.sqrt(6.0 / (16 + 32))
    assert np.all(np.abs(w) <= limit + 1e-5), "kernel values exceed xavier_uniform bound"


def test_patch_token_embedder_bias_zeros():
    m = PatchTokenEmbedder(16, 32, rngs=make_rngs(), param_dtype=jnp.float32)
    b = np.asarray(m.proj.bias.get_value())
    np.testing.assert_array_equal(b, 0.0, err_msg="bias must be zero-initialized")


# ---------------------------------------------------------------------------
# PixelTokenEmbedder
# ---------------------------------------------------------------------------


def test_pixel_token_embedder_output_shape():
    """Output shape: (B·L, p², D_pix)."""
    B, H, W, C = 2, 8, 8, 3
    p = 2
    Hs, Ws = H // p, W // p
    D_pix = 32
    m = PixelTokenEmbedder(
        in_channels=C, pixel_hidden_size=D_pix,
        field_shape=(H, W), patch_size=p,
        rngs=make_rngs(),
    )
    x = jnp.ones((B, H, W, C))
    y = m(x)
    assert y.shape == (B * Hs * Ws, p * p, D_pix)


def test_pixel_token_embedder_no_abs_pos():
    """use_abs_pos=False: all-ones input → all pixel tokens identical (no positional shift)."""
    B, H, W, C = 1, 4, 4, 2
    p = 2
    D_pix = 16
    m = PixelTokenEmbedder(C, D_pix, field_shape=(H, W), patch_size=p, use_abs_pos=False, rngs=make_rngs())
    x = jnp.ones((B, H, W, C))
    y = m(x)  # (B*L, p², D_pix)
    y_np = np.asarray(y)
    # Every pixel token must be identical since input is uniform and no pos is added.
    np.testing.assert_array_equal(
        y_np, np.broadcast_to(y_np[0:1, 0:1, :], y_np.shape),
        err_msg="use_abs_pos=False with uniform input must produce identical pixel tokens",
    )


def test_pixel_token_embedder_abs_pos_differs_from_no_pos():
    """use_abs_pos=True must produce a different result than use_abs_pos=False."""
    B, H, W, C = 1, 4, 4, 2
    p = 2
    D_pix = 16
    m_pos = PixelTokenEmbedder(C, D_pix, field_shape=(H, W), patch_size=p, use_abs_pos=True, rngs=make_rngs())
    m_no = PixelTokenEmbedder(C, D_pix, field_shape=(H, W), patch_size=p, use_abs_pos=False, rngs=make_rngs())
    x = jnp.ones((B, H, W, C))
    y_pos = m_pos(x)
    y_no = m_no(x)
    assert not np.allclose(np.asarray(y_pos), np.asarray(y_no)), (
        "use_abs_pos=True and False should produce different outputs"
    )


def test_pixel_token_embedder_pixel_grouping():
    """Pixel (i,j) within patch (a,b) must preserve spatial identity.

    After projection (but before abs-pos), the grouped pixel at position
    (patch_a * Ws + patch_b, i*p+j, :) should come from input pixel
    (a*p+i, b*p+j, :) projected by the linear.
    """
    p = 2
    H, W, C = 4, 4, 1
    D_pix = 8
    m = PixelTokenEmbedder(C, D_pix, field_shape=(H, W), patch_size=p, use_abs_pos=False, rngs=make_rngs(), param_dtype=jnp.float32)

    # Use identity-like inputs: each pixel is its own unique scalar
    x_np = np.arange(H * W, dtype=np.float32).reshape(1, H, W, 1)
    x = jnp.array(x_np)

    y = m(x)  # (B*L, p², D_pix)
    Hs = H // p
    Ws = W // p

    # Verify grouping by checking that the linear proj of each pixel is consistent
    # We compare m(single_pixel_hot) to check spatial ordering.
    # Simpler: project each pixel individually through the same linear and check.
    proj_all = m.proj(x)  # (B, H, W, D_pix) before grouping
    proj_np = np.asarray(proj_all)

    y_np = np.asarray(y)  # (B*L, p², D_pix)
    for a in range(Hs):
        for b_col in range(Ws):
            patch_idx = a * Ws + b_col
            for i in range(p):
                for j in range(p):
                    pix_in_patch = i * p + j
                    got = y_np[patch_idx, pix_in_patch, :]
                    expected = proj_np[0, a * p + i, b_col * p + j, :]
                    np.testing.assert_allclose(
                        got, expected, atol=1e-5,
                        err_msg=f"pixel grouping mismatch at patch ({a},{b_col}) pixel ({i},{j})",
                    )


def test_pixel_token_embedder_wrong_shape_raises():
    """Passing input with (H, W) != field_shape must raise ValueError."""
    H, W, C, p, D_pix = 4, 4, 2, 2, 16
    m = PixelTokenEmbedder(C, D_pix, field_shape=(H, W), patch_size=p, rngs=make_rngs())
    x_wrong = jnp.ones((1, H + 2, W, C))
    with pytest.raises(ValueError, match="field_shape"):
        m(x_wrong)


def test_pixel_token_embedder_pos_orientation_nonsquare():
    """Non-square grid: token at pixel (h, w) minus projected input equals sincos row h*W+w.

    Pins the (H*W, D) → (H, W, D) reshape orientation used for the pos table.
    """
    from gensbi.experimental.models.pixeldit.modules import get_2d_sincos_pos_embed

    H, W, C, p, D_pix = 4, 6, 1, 2, 32
    # dtype=jnp.float32 pins full-precision compute: this test verifies pos-
    # embedding orientation math against an exact fp32 reference table, not
    # mixed-precision behavior (bf16 compute would round entries visibly).
    m = PixelTokenEmbedder(
        C, D_pix, field_shape=(H, W), patch_size=p,
        use_abs_pos=True, rngs=make_rngs(), dtype=jnp.float32, param_dtype=jnp.float32,
    )
    m_no = PixelTokenEmbedder(
        C, D_pix, field_shape=(H, W), patch_size=p,
        use_abs_pos=False, rngs=make_rngs(), dtype=jnp.float32, param_dtype=jnp.float32,
    )
    # Share projection weights so we can isolate the pos contribution.
    m_no.proj = m.proj

    x = jnp.ones((1, H, W, C))
    y_pos = np.asarray(m(x))     # (B*L, p², D_pix) with pos
    y_no = np.asarray(m_no(x))   # same, without pos

    sincos = get_2d_sincos_pos_embed(D_pix, H, W)  # (H*W, D_pix) float32

    Hs, Ws = H // p, W // p
    for h in range(H):
        for w in range(W):
            # pixel (h, w) lives in patch (h//p, w//p) at local position (h%p, w%p)
            patch_idx = (h // p) * Ws + (w // p)
            pix_in_patch = (h % p) * p + (w % p)
            diff = y_pos[patch_idx, pix_in_patch, :] - y_no[patch_idx, pix_in_patch, :]
            expected = sincos[h * W + w].astype(np.float32)
            np.testing.assert_allclose(
                diff, expected, atol=1e-4,
                err_msg=f"pos orientation mismatch at pixel ({h},{w})",
            )


# ---------------------------------------------------------------------------
# CondTokenEmbedder
# ---------------------------------------------------------------------------


def test_cond_token_embedder_output_shape_3d():
    """(B, K, D_c) input → (B, K, D) output."""
    B, K, D_c, D = 2, 4, 8, 32
    m = CondTokenEmbedder(cond_in_channels=D_c, hidden_size=D, n_tokens=K, rngs=make_rngs())
    x = jnp.ones((B, K, D_c))
    y = m(x)
    assert y.shape == (B, K, D)


def test_cond_token_embedder_2d_input_expands():
    """(B, K) input with cond_in_channels=1 is auto-expanded to (B, K, 1)."""
    B, K, D = 2, 4, 32
    m = CondTokenEmbedder(cond_in_channels=1, hidden_size=D, n_tokens=K, rngs=make_rngs())
    x = jnp.ones((B, K))
    y = m(x)
    assert y.shape == (B, K, D)


def test_cond_token_embedder_2d_raises_wrong_channels():
    """(B, K) with cond_in_channels=2 must raise ValueError."""
    B, K, D = 2, 4, 32
    m = CondTokenEmbedder(cond_in_channels=2, hidden_size=D, n_tokens=K, rngs=make_rngs())
    with pytest.raises(ValueError):
        m(jnp.ones((B, K)))


def test_cond_token_embedder_none_vs_absolute_differ():
    """id_embedding='none' and 'absolute' must produce different outputs."""
    B, K, D_c, D = 2, 4, 8, 32
    m_abs = CondTokenEmbedder(D_c, D, K, id_embedding="absolute", rngs=make_rngs(), param_dtype=jnp.float32)
    m_none = CondTokenEmbedder(D_c, D, K, id_embedding="none", rngs=make_rngs(), param_dtype=jnp.float32)
    x = jnp.ones((B, K, D_c))
    y_abs = m_abs(x)
    y_none = m_none(x)
    assert not np.allclose(np.asarray(y_abs), np.asarray(y_none)), (
        "absolute id_embedding must shift the output relative to none"
    )


def test_cond_token_embedder_pos1d():
    """id_embedding='pos1d' must run without error and return correct shape."""
    B, K, D_c, D = 2, 6, 4, 32
    m = CondTokenEmbedder(D_c, D, K, id_embedding="pos1d", rngs=make_rngs())
    x = jnp.ones((B, K, D_c))
    y = m(x)
    assert y.shape == (B, K, D)
