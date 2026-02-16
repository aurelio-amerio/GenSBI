import pytest
import jax.numpy as jnp
from flax import nnx
import jax

from gensbi.models.flux1.layers import EmbedND, MLPEmbedder, Modulation

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
    # Modulation produces shift, scale, gate.
    # Each should have shape (Batch, 1, dim) because of:
    # out = jnp.split(self.lin(nnx.silu(vec))[:, None, :], self.multiplier, axis=-1)
    # self.lin outputs multiplier * dim.
    # [:, None, :] adds a dimension.

    assert out1.shift.shape == (2, 1, dim)
    assert out1.scale.shape == (2, 1, dim)
    assert out1.gate.shape == (2, 1, dim)

    assert out2.shift.shape == (2, 1, dim)
    assert out2.scale.shape == (2, 1, dim)
    assert out2.gate.shape == (2, 1, dim)

    # Verify zero initialization
    # Since kernel and bias are zeros, output should be zero.
    # But nnx.silu(vec) might not be zero.
    # However, linear layer with zero weights/bias produces zero output regardless of input.
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
