import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
import pytest

from gensbi.recipes.utils import healpix_rope_theta, init_ids_healpix


def test_healpix_rope_theta_follows_project_convention():
    # Project convention: theta = 10 * token count (Flux1Params defaults to
    # 10 * (dim_obs + dim_cond) at model.py:184). Full sky has 12*nside^2 tokens.
    assert healpix_rope_theta(2) == 480
    assert healpix_rope_theta(4) == 1920
    assert healpix_rope_theta(4) > healpix_rope_theta(2)
    assert isinstance(healpix_rope_theta(2), int)


# 1 / (pixel angular size) = nside * sqrt(3/pi); adjacent tokens ~1 apart.
def _pixel_unit_radius(nside):
    return nside * np.sqrt(3.0 / np.pi)


def test_init_ids_healpix_shape_dtype_scale():
    ids, n = init_ids_healpix(2)
    assert n == 48  # 12 * nside^2
    assert ids.shape == (1, 48, 3)
    assert ids.dtype == jnp.float32
    # every token direction lies on the sphere of radius r(nside)
    norms = np.linalg.norm(np.asarray(ids[0]), axis=-1)
    np.testing.assert_allclose(norms, _pixel_unit_radius(2), rtol=1e-5)


def test_init_ids_healpix_nest_order_roundtrip():
    # Token i must be the center of NEST pixel i: healpy round-trip catches
    # any ordering or indexing bug in the builder.
    import healpy as hp

    nside = 4
    ids, n = init_ids_healpix(nside)
    vecs = np.asarray(ids[0]) / _pixel_unit_radius(nside)
    pix = hp.vec2pix(nside, vecs[:, 0], vecs[:, 1], vecs[:, 2], nest=True)
    np.testing.assert_array_equal(pix, np.arange(n))


def test_init_ids_healpix_validates_nside():
    with pytest.raises(ValueError, match="power of 2"):
        init_ids_healpix(3)
    with pytest.raises(ValueError, match="power of 2"):
        init_ids_healpix(0)


def test_init_ids_healpix_base_pixel_subset_matches_full_sky():
    # Subset ids must be exactly the corresponding rows of the full-sky ids:
    # the encoding depends only on token directions, never on token count.
    nside = 2
    full, _ = init_ids_healpix(nside)
    subset, n_sub = init_ids_healpix(nside, base_pixels=[3, 7])
    assert n_sub == 2 * nside**2
    face_len = nside**2
    expected = jnp.concatenate(
        [
            full[:, 3 * face_len : 4 * face_len],
            full[:, 7 * face_len : 8 * face_len],
        ],
        axis=1,
    )
    np.testing.assert_array_equal(np.asarray(subset), np.asarray(expected))


def test_init_ids_healpix_rejects_bad_base_pixels():
    with pytest.raises(ValueError, match="base_pixels"):
        init_ids_healpix(2, base_pixels=[0, 12])
    with pytest.raises(ValueError, match="base_pixels"):
        init_ids_healpix(2, base_pixels=[1, 1])


def test_no_face_seam_discontinuity():
    # The failure documented for index-based RoPE on HEALPix (StereoRoPE,
    # arXiv:2606.31248) is a discontinuity across base-face boundaries. In
    # chord coordinates, grid-neighbor distances must be uniform across the
    # whole sphere — face boundaries and poles included. An index-seam bug
    # would make some neighbor pairs ~nside times farther than others.
    import healpy as hp

    nside = 4
    ids, n = init_ids_healpix(nside)
    coords = np.asarray(ids[0])
    neigh = hp.get_all_neighbours(nside, np.arange(n), nest=True)  # (8, n)
    dists = []
    for p in range(n):
        for q in neigh[:, p]:
            if q >= 0:
                dists.append(np.linalg.norm(coords[p] - coords[q]))
    dists = np.asarray(dists)
    # pixel units: neighbor spacing ~1 (sides) to ~sqrt(2) (diagonals), with
    # HEALPix shape distortion on top; no seam outliers anywhere.
    assert dists.max() / dists.min() < 4.0
    assert 0.5 < dists.mean() < 2.0
