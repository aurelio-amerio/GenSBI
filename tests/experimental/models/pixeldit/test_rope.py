import jax
import jax.numpy as jnp
import numpy as np

from gensbi.experimental.models.pixeldit.rope import (
    precompute_freqs_cis_1d,
    precompute_freqs_cis_2d,
)
from gensbi.models.flux1.math import apply_rope


def test_shape_and_dtype():
    rot2d = precompute_freqs_cis_2d(64, 4, 6)
    assert rot2d.shape == (1, 1, 24, 32, 2, 2)
    assert rot2d.dtype == jnp.float32

    rot1d = precompute_freqs_cis_1d(64, 5)
    assert rot1d.shape == (1, 1, 5, 32, 2, 2)
    assert rot1d.dtype == jnp.float32


def test_rotation_orthogonal():
    for rot in (precompute_freqs_cis_2d(64, 4, 6), precompute_freqs_cis_1d(64, 5)):
        blocks = np.asarray(rot).reshape(-1, 2, 2)
        for r in blocks:
            np.testing.assert_allclose(r @ r.T, np.eye(2), atol=1e-5)
            np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-5)


def _reference_2d_cis(dim, height, width, theta=10000.0, scale=16.0):
    """Reference math reimplemented with numpy complex (modules.py:132-145)."""
    x_pos = np.linspace(0, scale, width)
    y_pos = np.linspace(0, scale, height)
    yy, xx = np.meshgrid(y_pos, x_pos, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    freqs = 1.0 / (theta ** (np.arange(0, dim, 4)[: dim // 4] / dim))
    x_ang = np.outer(xx, freqs)
    y_ang = np.outer(yy, freqs)
    x_cis = np.exp(1j * x_ang)
    y_cis = np.exp(1j * y_ang)
    cis = np.concatenate([x_cis[..., None], y_cis[..., None]], axis=-1)
    return cis.reshape(height * width, -1)  # (N, dim/2)


def test_matches_complex_reference():
    dim, height, width = 64, 4, 6
    n = height * width

    cis = _reference_2d_cis(dim, height, width)  # (N, dim/2) complex

    key = jax.random.PRNGKey(0)
    # apply_rope consumes q as (B, H, L, dim); the table broadcasts over (B, H).
    q = jax.random.normal(key, (1, 1, n, dim))

    rot = precompute_freqs_cis_2d(dim, height, width)
    q_rot, _ = apply_rope(q, q, rot)
    q_rot = np.asarray(q_rot).reshape(n, dim)

    # complex reference: rotate consecutive pairs (q[2j], q[2j+1]) by cis[n, j]
    q_np = np.asarray(q).reshape(n, dim)
    q_complex = q_np[:, 0::2] + 1j * q_np[:, 1::2]  # (N, dim/2)
    out_complex = q_complex * cis
    expected = np.empty((n, dim))
    expected[:, 0::2] = out_complex.real
    expected[:, 1::2] = out_complex.imag

    np.testing.assert_allclose(q_rot, expected, atol=1e-5)


def test_native_grid_spacing():
    width = height = 16
    scale = 16.0
    head_dim = 64

    rot = precompute_freqs_cis_2d(head_dim, height, width, scale=scale)
    # Table is (1, 1, N, head_dim/2, 2, 2). Pair index 0 is the lowest x
    # frequency (freqs[0] = theta**0 = 1.0), so its angle for token (i, j) is
    # exactly x_pos[j]. Row i=0 (tokens j=0..width-1 at flat index j) recovers
    # the x-position spacing directly from the rotation matrices.
    row0 = np.asarray(rot)[0, 0, :width, 0]  # (width, 2, 2)

    # Compose R_{j+1} @ R_j^T to get the spacing rotation, then read its angle
    # via atan2 so the result is wrap-safe even though raw positions reach 16.
    spacings = []
    for j in range(width - 1):
        delta = row0[j + 1] @ row0[j].T
        spacings.append(np.arctan2(delta[1, 0], delta[0, 0]))
    spacings = np.asarray(spacings)

    np.testing.assert_allclose(spacings, scale / (width - 1), atol=1e-5)
    np.testing.assert_allclose(spacings, 16.0 / 15.0, atol=1e-5)
