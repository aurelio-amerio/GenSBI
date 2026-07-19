import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

from gensbi.recipes import HealpixRope, IdStrategy
from gensbi.recipes.utils import healpix_rope_theta, init_ids_healpix


def test_healpix_rope_build_matches_init_ids_healpix():
    ids, n = HealpixRope(nside=2).build(48)
    ref_ids, ref_n = init_ids_healpix(2)
    assert n == ref_n == 48
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))


def test_healpix_rope_subset_build_matches_init_ids_healpix():
    ids, n = HealpixRope(nside=2, base_pixels=(3, 7)).build(8)
    ref_ids, ref_n = init_ids_healpix(2, base_pixels=[3, 7])
    assert n == ref_n == 8
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))


def test_healpix_rope_dim_mismatch_names_both_numbers():
    # The error must name the given dim AND the expected token count.
    with pytest.raises(ValueError, match=r"(?s)47.*48|48.*47"):
        HealpixRope(nside=2).build(47)


def test_healpix_rope_theta_property():
    assert HealpixRope(nside=2).theta == healpix_rope_theta(2) == 480


def test_healpix_rope_num_tokens():
    assert HealpixRope(nside=2).num_tokens == 48
    assert HealpixRope(nside=2, base_pixels=(0, 1, 2)).num_tokens == 12


def test_healpix_rope_validates_at_construction():
    with pytest.raises(ValueError, match="power of 2"):
        HealpixRope(nside=3)
    with pytest.raises(ValueError, match="non-empty"):
        HealpixRope(nside=2, base_pixels=())
    with pytest.raises(ValueError, match="integer"):
        HealpixRope(nside=2, base_pixels=(1.5,))


def test_healpix_rope_satisfies_id_strategy_protocol():
    strat = HealpixRope(nside=2)
    assert isinstance(strat, IdStrategy)
    assert strat.name == "healpix-rope"
