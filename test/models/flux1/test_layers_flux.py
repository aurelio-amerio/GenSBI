import pytest
import jax.numpy as jnp
from flax import nnx
import jax

from gensbi.models.flux1.layers import (
    EmbedND, MLPEmbedder, Modulation, timestep_embedding, QKNorm, SelfAttention,
    DoubleStreamBlock, SingleStreamBlock, LastLayer
)

def get_rngs():
    return nnx.Rngs(0)

def test_embed_nd_shape():
    batch_size = 2
    seq_len = 5
    n_axes = 2
    dim = 16
    theta = 10000
    axes_dim = [8, 8]  # Sum is 16.

    # EmbedND expects axes_dim to be provided.
    # It calculates embeddings for each axis.
    # Each axis i gets an embedding of size axes_dim[i].
    # rope returns shape (..., axes_dim[i]/2, 2, 2).
    # EmbedND concatenates these along axis -3.
    # So the resulting dimension is sum(axes_dim)/2.

    layer = EmbedND(dim, theta, axes_dim)

    # ids shape: (Batch, SeqLen, n_axes)
    ids = jnp.zeros((batch_size, seq_len, n_axes), dtype=jnp.float32)

    output = layer(ids)

    # Expected output shape: (Batch, 1, SeqLen, sum(axes_dim)/2, 2, 2)
    expected_dim_sum = sum(axes_dim) // 2
    expected_shape = (batch_size, 1, seq_len, expected_dim_sum, 2, 2)

    assert output.shape == expected_shape
    assert output.dtype == jnp.float32

def test_embed_nd_mismatch_axes():
    # Test behavior when input axes count doesn't match initialization
    dim = 16
    theta = 10000
    axes_dim = [8, 8]
    layer = EmbedND(dim, theta, axes_dim)

    # Case 1: More axes in input than configured -> IndexError
    ids_too_many = jnp.zeros((1, 5, 3)) # 3 axes, but configured for 2
    with pytest.raises(IndexError):
        layer(ids_too_many)

    # Case 2: Fewer axes in input than configured -> subset of embeddings
    ids_too_few = jnp.zeros((1, 5, 1)) # 1 axis
    output = layer(ids_too_few)

    # Should only use the first axis_dim (8)
    # Output shape: (1, 1, 5, 8/2, 2, 2) -> (1, 1, 5, 4, 2, 2)
    expected_shape = (1, 1, 5, 4, 2, 2)
    assert output.shape == expected_shape


def test_mlp_embedder_shape():
    in_dim = 10
    hidden_dim = 20
    rngs = get_rngs()

    layer = MLPEmbedder(in_dim, hidden_dim, rngs=rngs)

    x = jnp.ones((2, in_dim)) # Batch size 2
    output = layer(x)

    assert output.shape == (2, hidden_dim)

    # Test with different dtype
    layer_bf16 = MLPEmbedder(in_dim, hidden_dim, rngs=rngs, param_dtype=jnp.bfloat16)
    assert layer_bf16.in_layer.kernel.dtype == jnp.bfloat16
    assert layer_bf16.out_layer.kernel.dtype == jnp.bfloat16

def test_modulation_shape():
    dim = 16
    rngs = get_rngs()

    # Test double=True
    mod_double = Modulation(dim, double=True, rngs=rngs)
    vec = jnp.ones((2, dim)) # Batch size 2
    out1, out2 = mod_double(vec)

    assert out1 is not None
    assert out2 is not None

    # Check shapes of ModulationOut components
    assert out1.shift.shape == (2, 1, dim)
    assert out1.scale.shape == (2, 1, dim)
    assert out1.gate.shape == (2, 1, dim)
    assert out2.shift.shape == (2, 1, dim)
    assert out2.scale.shape == (2, 1, dim)
    assert out2.gate.shape == (2, 1, dim)

    # Verify zero initialization
    assert jnp.all(out1.shift == 0)
    assert jnp.all(out1.scale == 0)
    assert jnp.all(out1.gate == 0)
    assert jnp.all(out2.shift == 0)
    assert jnp.all(out2.scale == 0)
    assert jnp.all(out2.gate == 0)

    # Test double=False
    mod_single = Modulation(dim, double=False, rngs=rngs)
    out1, out2 = mod_single(vec)

    assert out1 is not None
    assert out2 is None

    assert out1.shift.shape == (2, 1, dim)
    assert out1.scale.shape == (2, 1, dim)
    assert out1.gate.shape == (2, 1, dim)

def test_timestep_embedding_shape():
    batch_size = 4
    dim = 256
    t = jnp.ones((batch_size,))

    emb = timestep_embedding(t, dim)

    assert emb.shape == (batch_size, dim)
    # Check if dtype propagates correctly
    assert emb.dtype == t.dtype

def test_qk_norm_shape():
    dim = 16
    rngs = get_rngs()
    layer = QKNorm(dim, rngs=rngs)

    q = jnp.ones((2, 5, dim))
    k = jnp.ones((2, 5, dim))
    v = jnp.ones((2, 5, dim))

    q_norm, k_norm = layer(q, k, v)

    assert q_norm.shape == q.shape
    assert k_norm.shape == k.shape

def test_self_attention_shape():
    dim = 32
    num_heads = 4
    rngs = get_rngs()

    layer = SelfAttention(dim, rngs=rngs, num_heads=num_heads)

    x = jnp.ones((2, 5, dim)) # (B, L, D)

    out = layer(x, pe=None)
    assert out.shape == x.shape

def test_double_stream_block_shape():
    hidden_size = 32
    num_heads = 4
    mlp_ratio = 4.0
    rngs = get_rngs()

    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, rngs=rngs)

    obs = jnp.ones((2, 5, hidden_size)) # (B, L_obs, D)
    cond = jnp.ones((2, 3, hidden_size)) # (B, L_cond, D)
    vec = jnp.ones((2, hidden_size)) # (B, D)

    obs_out, cond_out = block(obs, cond, vec)

    assert obs_out.shape == obs.shape
    assert cond_out.shape == cond.shape

def test_single_stream_block_shape():
    hidden_size = 32
    num_heads = 4
    rngs = get_rngs()

    block = SingleStreamBlock(hidden_size, num_heads, rngs=rngs)

    x = jnp.ones((2, 8, hidden_size)) # (B, L, D)
    vec = jnp.ones((2, hidden_size)) # (B, D)

    out = block(x, vec)

    assert out.shape == x.shape

def test_last_layer_shape():
    hidden_size = 32
    patch_size = 1
    out_channels = 3
    rngs = get_rngs()

    layer = LastLayer(hidden_size, patch_size, out_channels, rngs=rngs)

    x = jnp.ones((2, 5, hidden_size))
    vec = jnp.ones((2, hidden_size))

    out = layer(x, vec)

    # Expected output: (B, L, patch_size^2 * out_channels)
    expected_shape = (2, 5, patch_size * patch_size * out_channels)

    assert out.shape == expected_shape
