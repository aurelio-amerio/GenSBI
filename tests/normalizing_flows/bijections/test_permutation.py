import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.permutation import Permutation


def test_reverse_permutation_roundtrip_and_zero_logdet():
    perm = Permutation.reverse(4)
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    u, logdet_inv = perm.inverse(x)
    assert jnp.array_equal(u, jnp.array([4.0, 3.0, 2.0, 1.0]))
    assert logdet_inv == 0.0
    x2, logdet_fwd = perm.forward(u)
    assert jnp.array_equal(x2, x)
    assert logdet_fwd == 0.0


def test_random_permutation_is_a_bijection():
    perm = Permutation.random(6, rngs=nnx.Rngs(0))
    x = jnp.arange(6.0)
    u, _ = perm.inverse(x)
    x2, _ = perm.forward(u)
    assert jnp.array_equal(x, x2)
    assert jnp.array_equal(jnp.sort(u), x)   # a true permutation
