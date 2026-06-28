import jax.numpy as jnp
import pytest
from gensbi.models.core.tokenizers import VectorTokenizer


def test_shapes_scalar_per_token():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert (tok.T, tok.F) == (6, 1)
    x = jnp.arange(12.0).reshape(2, 6)
    t = tok.tokenize(x)
    assert t.shape == (2, 6, 1)


def test_shapes_block_per_token():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert (tok.T, tok.F) == (3, 2)
    x = jnp.arange(12.0).reshape(2, 6)
    assert tok.tokenize(x).shape == (2, 3, 2)


def test_roundtrip_identity():
    tok = VectorTokenizer(dim=6, block_size=2)
    x = jnp.arange(12.0).reshape(2, 6)
    assert jnp.allclose(tok.detokenize(tok.tokenize(x)), x)


def test_block_size_must_divide_dim():
    with pytest.raises(ValueError):
        VectorTokenizer(dim=5, block_size=2)


import jax
from gensbi.models.core.tokenizers import ImageTokenizer
from gensbi.models.core.patching import patchify_2d


def test_vector_tokenizer_example_shape():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert tok.example_shape == (6,)


def test_image_tokenizer_shapes():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    assert (tok.T, tok.F) == (16, 8)          # T=(8/2)^2=16, F=2*2*2=8
    assert tok.example_shape == (8, 8, 2)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 8, 2))
    assert tok.tokenize(x).shape == (3, 16, 8)


def test_image_tokenizer_matches_patchify_2d():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 8, 2))
    assert jnp.allclose(tok.tokenize(x), patchify_2d(x, size=2))


def test_image_tokenizer_roundtrip():
    tok = ImageTokenizer(height=8, width=8, channels=2, patch_size=2)
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 8, 8, 2))
    assert jnp.allclose(tok.detokenize(tok.tokenize(x)), x, atol=1e-6)


def test_image_tokenizer_non_divisible_raises():
    with pytest.raises(ValueError):
        ImageTokenizer(height=7, width=8, channels=1, patch_size=2)


def test_vector_tokenizer_channels_one_unchanged():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert tok.F == 2 and tok.T == 3 and tok.example_shape == (6,)
    x = jnp.arange(2 * 6).reshape(2, 6).astype(jnp.float32)
    assert jnp.array_equal(tok.detokenize(tok.tokenize(x)), x)


def test_vector_tokenizer_channels_roundtrip():
    tok = VectorTokenizer(dim=6, block_size=2, channels=2)
    assert tok.F == 4 and tok.T == 3 and tok.example_shape == (6, 2)
    x = jnp.arange(2 * 6 * 2).reshape(2, 6, 2).astype(jnp.float32)
    z = tok.tokenize(x)
    assert z.shape == (2, 3, 4)
    assert jnp.array_equal(tok.detokenize(z), x)


def test_vector_tokenizer_channels_zero_raises():
    with pytest.raises(ValueError):
        VectorTokenizer(dim=6, channels=0)
