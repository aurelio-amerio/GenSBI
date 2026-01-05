import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

from gensbi.models.embedding.embedding import MLPEmbedder, GaussianFourierEmbedding, SimpleTimeEmbedding, SinusoidalTimeEmbedding

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
