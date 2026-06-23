import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner


def test_embed_shape_and_inject_broadcasts():
    cond_dim, channels, B, T = 3, 8, 4, 5
    c = VectorConditioner(cond_dim, channels, rngs=nnx.Rngs(0))
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim))
    sig = c.embed(cond)
    assert sig.shape == (B, channels)
    tokens = jnp.zeros((B, T, channels))
    out = c.inject(tokens, sig)
    assert out.shape == (B, T, channels)
    # same signal added to every token
    assert jnp.allclose(out[:, 0, :], sig)
    assert jnp.allclose(out[:, 0, :], out[:, T - 1, :])


def test_unconditional_passthrough():
    c = VectorConditioner(0, 8, rngs=nnx.Rngs(0))
    assert c.embed(jnp.zeros((4, 0))) is None
    tokens = jnp.ones((4, 5, 8))
    assert jnp.allclose(c.inject(tokens, None), tokens)


def test_missing_cond_raises():
    c = VectorConditioner(3, 8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        c.embed(None)
