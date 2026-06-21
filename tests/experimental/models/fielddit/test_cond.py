import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.experimental.models.fielddit.cond import ScalarCondEmbedder


def test_scalar_cond_embedder_tokens_and_summary():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 1))  # (B, k=3, c=1)
    tokens, summary = emb(cond)
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
    assert jnp.all(jnp.isfinite(tokens)) and jnp.all(jnp.isfinite(summary))


def test_scalar_cond_embedder_accepts_2d_input():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(0), (2, 3))  # (B, k) -> expanded to (B, k, 1)
    tokens, summary = emb(cond)
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
    assert jnp.all(jnp.isfinite(tokens)) and jnp.all(jnp.isfinite(summary))


def test_scalar_cond_embedder_rejects_2d_when_multichannel():
    """(B, k) shorthand is only valid for in_channels == 1."""
    emb = ScalarCondEmbedder(in_channels=2, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    cond_2d = jnp.ones((2, 3))
    with pytest.raises(ValueError, match="cond_in_channels"):
        emb(cond_2d)


def test_scalar_cond_embedder_accepts_2d_when_single_channel():
    emb = ScalarCondEmbedder(in_channels=1, hidden_size=16, rngs=nnx.Rngs(0), param_dtype=jnp.float32)
    tokens, summary = emb(jnp.ones((2, 3)))
    assert tokens.shape == (2, 3, 16)
    assert summary.shape == (2, 16)
