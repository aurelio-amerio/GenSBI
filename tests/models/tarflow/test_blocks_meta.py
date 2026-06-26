import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.models.tarflow.blocks import MetaBlock
from gensbi.models.tarflow.conditioners import VectorConditioner
from gensbi.models.tarflow.conditioners import VectorPrefixConditioner


def _make(T=4, F=1, channels=8, cond_dim=2, zero_init=True, rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)                     # identity perm
    inv_perm = jnp.argsort(perm)
    cond = VectorConditioner(cond_dim, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm, inv_perm=inv_perm,
                     conditioner=cond, num_layers=2, num_heads=2, expansion=2,
                     rngs=rngs, zero_init=zero_init)


def test_zero_init_is_identity():
    blk = _make(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(2), (3, 2))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)


def test_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i (F=1 ⇒ clean (T,T) Jacobian)."""
    blk = _make(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])     # (1, T, 1)
        return z[0, :, 0]                            # (T,)

    J = jax.jacrev(f)(x0[:, 0])                      # (T, T)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_inverse_logdet_matches_autodiff():
    blk = _make(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_forward_inverse_roundtrip():
    blk = _make(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)


def _make_prefix(T=4, F=1, channels=8, cond_dim=2, num_tokens=2, zero_init=False,
                 rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)
    cond = VectorPrefixConditioner(cond_dim, channels, num_tokens, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     inv_perm=jnp.argsort(perm), conditioner=cond, num_layers=2,
                     num_heads=2, expansion=2, rngs=rngs, zero_init=zero_init)


def test_prefix_inverse_is_triangular():
    """z[i] must not depend on x[j] for j > i, with a prefix condition."""
    blk = _make_prefix(F=1, zero_init=False)
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(3), (T, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    J = jax.jacrev(f)(x0[:, 0])
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, j], 0.0, atol=1e-6), (i, j)


def test_prefix_logdet_matches_autodiff():
    blk = _make_prefix(F=1, zero_init=False)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (4, 1))
    cond = jnp.array([0.3, -0.4])

    def f(x):
        z, _ = blk.inverse(x[None], cond[None])
        return z[0, :, 0]

    _, ad = jnp.linalg.slogdet(jax.jacobian(f)(x0[:, 0]))
    _, analytic = blk.inverse(x0[None], cond[None])
    assert jnp.allclose(ad, analytic[0], atol=1e-4)


def test_prefix_roundtrip():
    blk = _make_prefix(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    z, _ = blk.inverse(x, cond)
    x_rt, _ = blk.forward(z, cond)
    assert jnp.allclose(x_rt, x, atol=1e-4)


def test_prefix_zero_init_identity():
    blk = _make_prefix(zero_init=True)
    x = jax.random.normal(jax.random.PRNGKey(7), (3, 4, 1))
    cond = jax.random.normal(jax.random.PRNGKey(8), (3, 2))
    z, logdet = blk.inverse(x, cond)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6)


def test_prefix_conditions_output():
    """The prefix must actually condition (a,b): different conditions must give
    different z. Guards against a regression that silently ignores the prefix."""
    blk = _make_prefix(F=1, zero_init=False)
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 4, 1))
    cond1 = jnp.broadcast_to(jnp.array([0.3, -0.4]), (2, 2))
    cond2 = jnp.broadcast_to(jnp.array([-0.7, 0.9]), (2, 2))
    z1, _ = blk.inverse(x, cond1)
    z2, _ = blk.inverse(x, cond2)
    assert not jnp.allclose(z1, z2, atol=1e-6)
