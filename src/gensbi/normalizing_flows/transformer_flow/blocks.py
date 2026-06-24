"""Transformer blocks for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.
Prefix-concatenation conditioning and SOS shift adapted from apple/ml-starflow
(STARFlow); see transformer_flow/LICENSE.starflow.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.normalizing_flows.bijections.base import Mask


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

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, hd)
        if mask is None:
            attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = jax.nn.dot_product_attention(q, k, v, mask=mask[None, None])
        x = x + self.proj(attn.reshape(B, T, C))
        h = self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))
        return x + h


class MetaBlock(nnx.Module):
    """One exact autoregressive bijection over a token sequence.

    inverse (data→noise): per-token affine ``z = (x − b)·exp(−a)`` with
    ``(a, b)`` produced by causal attention + shift-by-one, so token i's params
    depend only on tokens < i ⇒ triangular Jacobian, ``logdet = −Σ a``.
    forward (noise→data): sequential scan re-running the causal pass on the
    partially-built sequence (mirrors ``MaskedAutoregressive.forward``).
    """

    def __init__(self, F, channels, T, perm, inv_perm, conditioner,
                 num_layers, head_dim, expansion, rngs, zero_init=True):
        self.F = F
        self.T = T
        self.perm = Mask(jnp.asarray(perm, dtype=jnp.int32))
        self.inv_perm = Mask(jnp.asarray(inv_perm, dtype=jnp.int32))
        self.conditioner = conditioner
        self.proj_in = nnx.Linear(F, channels, rngs=rngs)
        self.sos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, F)) * 1e-2)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (T, channels)) * 1e-2)
        self.attn_blocks = nnx.List(
            [AttentionBlock(channels, head_dim, expansion, rngs)
             for _ in range(num_layers)])
        self.proj_out = nnx.Linear(channels, 2 * F, rngs=rngs)
        if zero_init:
            self.proj_out.kernel[...] = jnp.zeros_like(self.proj_out.kernel[...])
            self.proj_out.bias[...] = jnp.zeros_like(self.proj_out.bias[...])

    def _params(self, x_perm: Array, cond: Array | None):
        """(a, b) for the permuted tokens. SOS input-shift makes token i's
        params depend only on tokens < i (and the condition)."""
        bias, prefix = self.conditioner.embed(cond)
        B = x_perm.shape[0]
        sos = jnp.broadcast_to(self.sos_embed[...], (B, 1, self.F))
        x_in = jnp.concatenate([sos, x_perm[:, :-1]], axis=1)   # (B, T, F)
        h = self.proj_in(x_in) + self.pos_embed[...][None]      # (B, T, C)
        if bias is not None:
            h = h + bias[:, None, :]
        for blk in self.attn_blocks:
            h = blk(h)                                          # is_causal
        out = self.proj_out(h)                                  # (B, T, 2F)
        a, b = jnp.split(out, 2, axis=-1)                       # each (B, T, F)
        return a, b

    def inverse(self, x: Array, cond: Array | None = None):
        x = x.reshape(x.shape[0], self.T, self.F)
        xp = x[:, self.perm[...]]
        a, b = self._params(xp, cond)
        z = (xp - b) * jnp.exp(-a)
        logdet = -jnp.sum(a, axis=(1, 2))                      # (B,)
        z = z[:, self.inv_perm[...]]
        return z, logdet

    def forward(self, z: Array, cond: Array | None = None):
        z = z.reshape(z.shape[0], self.T, self.F)
        zp = z[:, self.perm[...]]

        def body(x, i):
            a, b = self._params(x, cond)        # a[:,i],b[:,i] depend on tokens < i
            xi = zp[:, i, :] * jnp.exp(a[:, i, :]) + b[:, i, :]
            return x.at[:, i, :].set(xi), None

        x = jnp.zeros_like(zp)
        x, _ = jax.lax.scan(body, x, jnp.arange(self.T))
        a, _ = self._params(x, cond)
        logdet = jnp.sum(a, axis=(1, 2))                       # (B,), +Σa
        x = x[:, self.inv_perm[...]]
        return x, logdet
