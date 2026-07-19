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


import grain
import jax
from flax import nnx  # noqa: F401  (parity with sibling test imports)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mock_models import MockConditionalModel

from gensbi.core import FlowMatchingMethod
from gensbi.recipes import ConditionalPipeline
from gensbi.recipes.utils import _resolve_embedding_ids, init_ids_1d


def _tiny_datasets(dim_obs, dim_cond, n=64, batch=8):
    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(key, (n, dim_obs, 1))
    x = jax.random.normal(key, (n, dim_cond, 1))

    def make(sl):
        data = np.concatenate([np.asarray(theta[sl]), np.asarray(x[sl])], axis=1)
        return (
            grain.MapDataset.source(data)
            .repeat()
            .to_iter_dataset()
            .batch(batch)
            .map(lambda d: (d[:, :dim_obs], d[:, dim_obs:]))
        )

    return make(slice(0, n // 2)), make(slice(n // 2, n))


def test_conditional_pipeline_accepts_healpix_rope_strategy():
    train_ds, val_ds = _tiny_datasets(3, 48)
    pipeline = ConditionalPipeline(
        model=MockConditionalModel(),
        train_dataset=train_ds,
        val_dataset=val_ds,
        dim_obs=3,
        dim_cond=48,
        method=FlowMatchingMethod(),
        id_embedding_strategy=("absolute", HealpixRope(nside=2)),
    )
    ref_ids, _ = init_ids_healpix(2)
    np.testing.assert_array_equal(np.asarray(pipeline.cond_ids), np.asarray(ref_ids))
    assert pipeline.dim_cond == 48
    # obs stream untouched: 1D absolute ids as before
    obs_ref, _ = init_ids_1d(3, semantic_id=0)
    np.testing.assert_array_equal(np.asarray(pipeline.obs_ids), np.asarray(obs_ref))


def test_conditional_pipeline_healpix_rope_dim_mismatch_raises():
    train_ds, val_ds = _tiny_datasets(3, 47)
    with pytest.raises(ValueError, match="healpix-rope"):
        ConditionalPipeline(
            model=MockConditionalModel(),
            train_dataset=train_ds,
            val_dataset=val_ds,
            dim_obs=3,
            dim_cond=47,
            method=FlowMatchingMethod(),
            id_embedding_strategy=("absolute", HealpixRope(nside=2)),
        )


def test_resolve_embedding_ids_dispatches_to_strategy_objects():
    ids, n = _resolve_embedding_ids(48, HealpixRope(nside=2), semantic_id=1)
    ref_ids, _ = init_ids_healpix(2)
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))
    assert n == 48


def test_resolve_embedding_ids_unknown_string_mentions_objects():
    # Prefix preserved for backward compat; message now teaches the object API.
    with pytest.raises(ValueError, match="Unknown id embedding strategy"):
        _resolve_embedding_ids(10, "rope3d", semantic_id=1)
    with pytest.raises(ValueError, match="IdStrategy"):
        _resolve_embedding_ids(10, "rope3d", semantic_id=1)
