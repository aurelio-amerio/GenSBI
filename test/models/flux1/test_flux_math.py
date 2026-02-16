import pytest
import jax.numpy as jnp
from gensbi.models.flux1.math import attention, rope, apply_rope

def test_rope_shape():
    B, L, D = 2, 10, 16
    pos = jnp.zeros((B, L))
    theta = 10000

    # Check assertion
    with pytest.raises(AssertionError):
        rope(pos, 15, theta)

    out = rope(pos, D, theta)
    # Expected shape: (B, L, D//2, 2, 2)
    assert out.shape == (B, L, D//2, 2, 2)

def test_rope_values():
    B, L, D = 1, 1, 4
    pos = jnp.zeros((B, L))
    theta = 100
    out = rope(pos, D, theta)
    # For pos=0, omega*pos = 0.
    # cos(0)=1, sin(0)=0.
    # out stack: [1, 0, 0, 1]
    # rearrange "b n d (i j) -> b n d i j", i=2, j=2
    # [1, 0] -> i=0. [0, 1] -> i=1.
    # d=D//2=2.
    # out[0, 0, :, 0, 0] should be 1
    # out[0, 0, :, 0, 1] should be 0
    # out[0, 0, :, 1, 0] should be 0
    # out[0, 0, :, 1, 1] should be 1

    expected = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    for d in range(D//2):
        assert jnp.allclose(out[0, 0, d], expected)

def test_apply_rope_shape():
    B, H, L, D = 2, 4, 10, 16
    q = jnp.ones((B, H, L, D))
    k = jnp.ones((B, H, L, D))

    # Create fake freqs_cis with correct shape for broadcasting
    # Shape from EmbedND: (B, 1, L, D//2, 2, 2)
    freqs_cis = jnp.ones((B, 1, L, D//2, 2, 2))

    q_out, k_out = apply_rope(q, k, freqs_cis)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype

def test_apply_rope_rotation():
    # Test that apply_rope actually rotates
    B, H, L, D = 1, 1, 1, 2
    q = jnp.array([[[[1.0, 0.0]]]]) # (1, 1, 1, 2)
    k = jnp.array([[[[1.0, 0.0]]]])

    # Rotation by 90 degrees (pi/2)
    # cos(90) = 0, sin(90) = 1
    # rope output structure: [[cos, -sin], [sin, cos]]
    # [[0, -1], [1, 0]]
    freqs_cis = jnp.array([[[[[[0.0, -1.0], [1.0, 0.0]]]]]]) # (1, 1, 1, 1, 2, 2)

    q_out, k_out = apply_rope(q, k, freqs_cis)

    # For 90 deg: c=0, s=1.
    # out = [0, 1]*1 + [-1, 0]*0 = [0, 1].
    expected = jnp.array([[[[0.0, 1.0]]]])
    assert jnp.allclose(q_out, expected, atol=1e-5)

def test_attention_shape():
    B, H, L, D = 2, 4, 10, 16
    q = jnp.ones((B, H, L, D))
    k = jnp.ones((B, H, L, D))
    v = jnp.ones((B, H, L, D))

    out = attention(q, k, v)
    # Output: B L (H D)
    assert out.shape == (B, L, H*D)

def test_attention_mask():
    B, H, L, D = 1, 1, 2, 2
    q = jnp.ones((B, H, L, D))
    k = jnp.ones((B, H, L, D))

    # v has distinct values for each token
    # Token 0: [10, 10]
    # Token 1: [20, 20]
    v = jnp.array([[[[10.0, 10.0], [20.0, 20.0]]]])

    # Mask: (B, H, L, L). Attend only to first token (index 0)
    # Token 0 attends to Token 0. Token 1 attends to Token 0.
    mask = jnp.array([[[[True, False], [True, False]]]])

    # Run with mask
    out = attention(q, k, v, mask=mask)
    assert out.shape == (B, L, H*D)

    # Expected output: both tokens attend to Token 0, so value is 10.
    assert jnp.allclose(out, 10.0)

def test_attention_pe():
    B, H, L, D = 2, 4, 10, 16
    q = jnp.ones((B, H, L, D))
    k = jnp.ones((B, H, L, D))
    v = jnp.ones((B, H, L, D))
    pe = jnp.ones((B, 1, L, D//2, 2, 2))

    out = attention(q, k, v, pe=pe)
    assert out.shape == (B, L, H*D)
