import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import MAFlow, MAFlowParams
from gensbi.normalizing_flows.bijections.transformers import Affine


def test_params_defaults_and_validation():
    p = MAFlowParams(rngs=nnx.Rngs(0), dim=3)
    assert isinstance(p.transformer, Affine)            # None -> Affine()
    assert p.cond_dim == 0 and p.n_layers == 5
    with pytest.raises(ValueError):
        MAFlowParams(rngs=nnx.Rngs(0), dim=3, permutation="nope")


def test_maflow_sample_shape_and_standardize():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3))
    c = jax.random.normal(jax.random.key(0), (7, 2, 1))
    assert flow.sample(jax.random.key(1), cond=c).shape == (7, 4, 1)
    assert flow.log_prob(jnp.zeros((5, 4, 1)), jnp.zeros((5, 2, 1))).shape == (5,)
    flow.set_standardization(jnp.ones((4, 1)), 2.0 * jnp.ones((4, 1)))
