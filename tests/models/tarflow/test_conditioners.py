import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from gensbi.models.tarflow.conditioners import VectorConditioner


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


from gensbi.models.tarflow.conditioners import (
    VectorPrefixConditioner, ImagePrefixConditioner,
)


def test_vector_prefix_shapes():
    c = VectorPrefixConditioner(cond_dim=3, channels=8, num_tokens=2, rngs=nnx.Rngs(0))
    assert c.M == 2
    cond = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    bias, prefix = c.embed(cond)
    assert bias is None
    assert prefix.shape == (4, 2, 8)


def test_image_prefix_shapes():
    # cond image 8x8x2, patch 2 -> M = 16 tokens
    c = ImagePrefixConditioner(cond_channels=2, patch_size=2, channels=8,
                               num_tokens=16, rngs=nnx.Rngs(0))
    assert c.M == 16
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 8, 8, 2))
    bias, prefix = c.embed(cond)
    assert bias is None
    assert prefix.shape == (4, 16, 8)


def test_prefix_depends_on_condition():
    c = VectorPrefixConditioner(cond_dim=3, channels=8, num_tokens=1, rngs=nnx.Rngs(0))
    _, p1 = c.embed(jnp.zeros((2, 3)))
    _, p2 = c.embed(jnp.ones((2, 3)))
    assert not jnp.allclose(p1, p2)
