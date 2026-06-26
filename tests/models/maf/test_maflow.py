import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import MAFlow, MAFlowParams
from gensbi.normalizing_flows import make_maf          # still present in Task 3
from gensbi.normalizing_flows.bijections.transformers import Affine


def test_params_defaults_and_validation():
    p = MAFlowParams(rngs=nnx.Rngs(0), dim=3)
    assert isinstance(p.transformer, Affine)            # None -> Affine()
    assert p.cond_dim == 0 and p.n_layers == 5
    with pytest.raises(ValueError):
        MAFlowParams(rngs=nnx.Rngs(0), dim=3, permutation="nope")


def test_maflow_matches_make_maf():
    cfg = dict(dim=3, cond_dim=2, n_layers=3, nn_width=16, nn_depth=2)
    old = make_maf(nnx.Rngs(0), **cfg)
    new = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), **cfg))
    x = jax.random.normal(jax.random.key(1), (5, 3))
    c = jax.random.normal(jax.random.key(2), (5, 2))
    assert jnp.allclose(old.log_prob(x, c), new.log_prob(x, c), atol=1e-5)


def test_maflow_sample_shape_and_standardize():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3))
    c = jax.random.normal(jax.random.key(0), (7, 2))
    s = flow.sample(jax.random.key(1), cond=c)
    assert s.shape == (7, 4)
    flow.set_standardization(jnp.ones(4), 2.0 * jnp.ones(4))   # no raise
