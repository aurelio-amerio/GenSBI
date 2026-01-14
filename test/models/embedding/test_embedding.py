import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.models.embedding import (
    MLPEmbedder,
    GaussianFourierEmbedding,
    SimpleTimeEmbedding,
    SinusoidalTimeEmbedding,
    SinusoidalPosEmbed1D,
    SinusoidalPosEmbed2D,
    Embed,
    FeatureEmbedder,
)


def get_rngs():
    return nnx.Rngs(0)


def test_mlp_embedder_output_shape():
    rngs = get_rngs()
    embedder = MLPEmbedder(in_dim=1, hidden_dim=4, rngs=rngs)
    x = jnp.ones((2, 3, 1))
    out = embedder(x)
    assert out.shape == (2, 3, 4)


def test_gaussian_fourier_embedding_output_shape():
    rngs = get_rngs()
    emb = GaussianFourierEmbedding(output_dim=8, rngs=rngs)
    t = jnp.ones((5, 1))
    out = emb(t)
    assert out.shape == (5, 8)


def test_simple_time_embedding_output_shape():
    emb = SimpleTimeEmbedding()
    t = jnp.ones((7, 1))
    out = emb(t)
    assert out.shape == (7, 4)


def test_sinusoidal_embedding_output_shape():
    emb = SinusoidalTimeEmbedding(output_dim=16)
    t = jnp.ones((10, 1))
    out = emb(t)
    assert out.shape == (10, 16)


def test_sinusoidal_pos_embed_1d_output_shape():
    emb = SinusoidalPosEmbed1D(hidden_size=12, max_len=50)
    ids = jnp.arange(0, 20).reshape(1, -1, 1)
    out = emb(ids)
    assert out.shape == (1, 20, 12)


def test_sinusoidal_pos_embed_2d_output_shape():
    emb = SinusoidalPosEmbed2D(hidden_size=16, max_h=100, max_w=101)
    ids = jnp.arange(0, 20).reshape(1, 5, 4)
    out = emb(ids)
    assert out.shape == (1, 20, 16)


def test_embed_output_shape():
    emb = Embed(num_embeddings=10, features=6, rngs=nnx.Rngs(0))
    ids = jnp.arange(0, 5).reshape(1, 5, 1)
    out = emb(ids)
    assert out.shape == (1, 5, 6)


def test_feature_embedder():
    absolute = FeatureEmbedder(5, 6, kind="absolute", rngs=nnx.Rngs(0))
    assert isinstance(absolute.embedder, Embed)

    pos1d = FeatureEmbedder(5, 6, kind="pos1d", rngs=nnx.Rngs(0))
    assert isinstance(pos1d.embedder, SinusoidalPosEmbed1D)

    pos2d = FeatureEmbedder(5, 6, kind="pos2d", rngs=nnx.Rngs(0))
    assert isinstance(pos2d.embedder, SinusoidalPosEmbed2D)

    # test value error
    with pytest.raises(ValueError):
        wrong = FeatureEmbedder(5, 6, kind="wrong", rngs=nnx.Rngs(0))
