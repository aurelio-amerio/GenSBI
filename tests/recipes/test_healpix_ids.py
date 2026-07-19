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
