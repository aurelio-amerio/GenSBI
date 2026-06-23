"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


class AttentionBlock(nnx.Module):
    """Pre-norm residual block: causal self-attention + MLP.

    LayerNorm is over the channel axis only (never across tokens), so it does
    not leak future tokens into earlier ones.
    """

    def __init__(self, channels: int, head_dim: int, expansion: int,
                 rngs: nnx.Rngs):
        if channels % head_dim != 0:
            raise ValueError(
                f"channels ({channels}) must be a multiple of head_dim "
                f"({head_dim})")
        self.num_heads = channels // head_dim
        self.head_dim = head_dim
        self.norm1 = nnx.LayerNorm(channels, rngs=rngs)
        self.qkv = nnx.Linear(channels, 3 * channels, rngs=rngs)
        self.proj = nnx.Linear(channels, channels, rngs=rngs)
        self.norm2 = nnx.LayerNorm(channels, rngs=rngs)
        self.mlp_in = nnx.Linear(channels, channels * expansion, rngs=rngs)
        self.mlp_out = nnx.Linear(channels * expansion, channels, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, hd)
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.reshape(B, T, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h
