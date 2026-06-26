"""Conditioning seams for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see models/tarflow/LICENSE.apple.
Prefix-concatenation conditioning adapted from apple/ml-starflow (STARFlow); see models/tarflow/LICENSE.starflow.

v1 ``VectorConditioner`` is the continuous analog of TarFlow's ``class_embed``:
an MLP embeds the condition to a ``channels``-vector that is broadcast-added to
every token. The signal depends only on the condition (constant w.r.t. the
modeled variable), so it shifts the affine params without breaking the
triangular Jacobian. A plain 2-layer MLP is used (not ``MLPEmbedder``, whose
``hidden_dim % in_dim == 0`` constraint does not fit arbitrary ``cond_dim``).
"""

import jax
from flax import nnx
from jax import Array

from gensbi.models.core.patching import patchify_2d


class VectorConditioner(nnx.Module):
    """MLP(cond) → per-token additive bias. ``cond_dim == 0`` ⇒ unconditional."""

    def __init__(self, cond_dim: int, channels: int, rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond: Array | None):
        """Return ``(bias, prefix)``; VectorConditioner only sets ``bias``."""
        if self.cond_dim == 0:
            return (None, None)
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        bias = self.l2(jax.nn.silu(self.l1(cond)))
        return (bias, None)


class VectorPrefixConditioner(nnx.Module):
    """Vector condition → ``num_tokens`` prefix tokens ``(B, M, channels)``."""

    def __init__(self, cond_dim: int, channels: int, num_tokens: int, rngs: nnx.Rngs):
        self.channels = channels
        self.M = num_tokens
        self.proj = nnx.Linear(cond_dim, channels * num_tokens, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, channels)) * 1e-2)

    def embed(self, cond: Array | None):
        if cond is None:
            raise ValueError("cond is required for VectorPrefixConditioner")
        B = cond.shape[0]
        h = self.proj(cond).reshape(B, self.M, self.channels)
        return (None, h + self.pos[...][None])


class ImagePrefixConditioner(nnx.Module):
    """Image condition ``(B, H, W, C)`` → ``M = (H/p)(W/p)`` prefix tokens."""

    def __init__(self, cond_channels: int, patch_size: int, channels: int,
                 num_tokens: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.channels = channels
        self.M = num_tokens
        in_f = cond_channels * patch_size * patch_size
        self.proj = nnx.Linear(in_f, channels, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, channels)) * 1e-2)

    def embed(self, cond: Array | None):
        if cond is None:
            raise ValueError("cond is required for ImagePrefixConditioner")
        patches = patchify_2d(cond, size=self.patch_size)      # (B, M, in_f)
        return (None, self.proj(patches) + self.pos[...][None])
