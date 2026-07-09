import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.models.maf.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.transformers import Affine


def _chain(dim=4, cond_dim=2, seed=0):
    rngs = nnx.Rngs(seed)
    bijections = [
        MaskedAutoregressive(dim, cond_dim, Affine(), 32, 2, rngs, zero_init=False),
        Permutation.reverse(dim),
        MaskedAutoregressive(dim, cond_dim, Affine(), 32, 2, rngs, zero_init=False),
    ]
    return Chain(bijections)


def test_chain_invertibility():
    chain = _chain()
    cond = jnp.array([0.2, -0.1])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])
    u, _ = chain.inverse(x, cond)
    x2, _ = chain.forward(u, cond)
    assert jnp.allclose(x, x2, atol=1e-5)


def test_chain_logdet_matches_autodiff():
    chain = _chain(dim=4)
    cond = jnp.array([0.2, -0.1])
    x = jnp.array([0.5, -1.0, 0.3, 0.8])

    def inv_only(x):
        return chain.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = chain.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)
