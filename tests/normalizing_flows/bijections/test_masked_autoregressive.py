import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.transformers import Affine


def _ma(dim=5, cond_dim=3, seed=0):
    return MaskedAutoregressive(
        dim=dim, cond_dim=cond_dim, transformer=Affine(),
        nn_width=32, nn_depth=2, zero_init=False, rngs=nnx.Rngs(seed),
    )


def test_invertibility_both_ways():
    ma = _ma()
    cond = jnp.array([0.1, -0.2, 0.3])
    x = jnp.array([0.5, -1.0, 0.2, 1.3, -0.7])
    u, _ = ma.inverse(x, cond)
    x2, _ = ma.forward(u, cond)
    assert jnp.allclose(x, x2, atol=1e-5)
    u2, _ = ma.inverse(x2, cond)
    assert jnp.allclose(u, u2, atol=1e-5)


def test_logdet_matches_autodiff_jacobian():
    """Spec §11 #3 — the sign/convention guardrail."""
    ma = _ma(dim=5)
    cond = jnp.array([0.1, -0.2, 0.3])
    x = jnp.array([0.5, -1.0, 0.2, 1.3, -0.7])

    def inv_only(x):
        return ma.inverse(x, cond)[0]

    _, ad_logdet = jnp.linalg.slogdet(jax.jacobian(inv_only)(x))
    _, analytic_logdet = ma.inverse(x, cond)
    assert jnp.allclose(ad_logdet, analytic_logdet, atol=1e-4)
