import jax.numpy as jnp
import pytest
from gensbi.normalizing_flows.transformer_flow.tokenizers import VectorTokenizer


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
