import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from gensbi.normalizing_flows.transformer_flow.conditioners import VectorConditioner


def test_embed_returns_bias_prefix_tuple():
    cond_dim, channels, B = 3, 8, 4
    c = VectorConditioner(cond_dim, channels, rngs=nnx.Rngs(0))
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim))
    bias, prefix = c.embed(cond)
    assert bias.shape == (B, channels)
    assert prefix is None


def test_unconditional_returns_none_none():
    c = VectorConditioner(0, 8, rngs=nnx.Rngs(0))
    assert c.embed(jnp.zeros((4, 0))) == (None, None)


def test_missing_cond_raises():
    c = VectorConditioner(3, 8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        c.embed(None)
