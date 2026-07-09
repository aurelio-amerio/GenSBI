import jax.numpy as jnp
from gensbi.normalizing_flows.bijections.standardize import Standardize


def test_default_is_identity():
    s = Standardize(dim=3)
    x = jnp.array([1.0, -2.0, 0.5])
    u, logdet = s.inverse(x)
    assert jnp.allclose(u, x)
    assert jnp.allclose(logdet, 0.0)


def test_standardize_roundtrip_and_logdet():
    s = Standardize(dim=3)
    s.set_stats(mean=jnp.array([1.0, 2.0, 3.0]), std=jnp.array([2.0, 0.5, 4.0]))
    x = jnp.array([3.0, 2.5, -1.0])
    u, logdet_inv = s.inverse(x)        # u = (x - mean) / std
    assert jnp.allclose(u, jnp.array([(3-1)/2, (2.5-2)/0.5, (-1-3)/4]))
    assert jnp.allclose(logdet_inv, -jnp.sum(jnp.log(jnp.array([2.0, 0.5, 4.0]))))
    x2, logdet_fwd = s.forward(u)
    assert jnp.allclose(x2, x, atol=1e-6)
    assert jnp.allclose(logdet_fwd, -logdet_inv)
