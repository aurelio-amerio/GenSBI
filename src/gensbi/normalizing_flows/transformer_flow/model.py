"""TransformerFlow: a transformer autoregressive normalizing flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

Batched-native ``(B, T, F)`` density model, sibling to the per-example
``Flow``/``Chain`` track. ``log_prob`` runs the parallel data→noise pass;
``sample`` runs the sequential noise→data pass. Base is a fixed ``N(0, I)`` over
the tokens (nvp mode).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.normalizing_flows.bijections.base import Mask
from gensbi.normalizing_flows.transformer_flow.blocks import MetaBlock
from gensbi.normalizing_flows.transformer_flow.conditioners import (
    VectorConditioner, VectorPrefixConditioner, ImagePrefixConditioner,
)
from gensbi.normalizing_flows.transformer_flow.tokenizers import (
    VectorTokenizer, ImageTokenizer,
)

_LOG2PI = jnp.log(2.0 * jnp.pi)


class TransformerFlow(nnx.Module):
    """Stack of MetaBlocks + tokenizer + standardization + N(0,I) base."""

    def __init__(self, blocks, tokenizer, dim, cond_dim, standardize=True):
        self.blocks = nnx.List(blocks)
        self.tokenizer = tokenizer
        self.dim = dim
        self.cond_dim = cond_dim
        self.T = tokenizer.T
        self.F = tokenizer.F
        self.example_shape = tokenizer.example_shape
        self._standardize = standardize
        self.mean = Mask(jnp.zeros(self.example_shape))
        self.std = Mask(jnp.ones(self.example_shape))

    def _base_log_prob(self, z: Array) -> Array:
        # z: (B, T, F); standard normal over (T, F)
        return -0.5 * jnp.sum(z ** 2, axis=(1, 2)) - 0.5 * self.T * self.F * _LOG2PI

    def _ensure_batched(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == len(self.example_shape):
            x = x[None]
        return x

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        x = self._ensure_batched(x)
        u = (x - self.mean[...]) / self.std[...]              # standardize
        logdet = -jnp.sum(jnp.log(self.std[...]))            # over all elements
        z = self.tokenizer.tokenize(u)                       # (B, T, F)
        total = jnp.broadcast_to(logdet, (x.shape[0],))
        for blk in self.blocks:
            z, ld = blk.inverse(z, cond)
            total = total + ld
        return self._base_log_prob(z) + total

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None):
        if cond is not None:
            nsamples = cond.shape[0]
        z = jax.random.normal(key, (nsamples, self.T, self.F))
        x = z
        for blk in reversed(self.blocks):
            x, _ = blk.forward(x, cond)
        x = self.tokenizer.detokenize(x)                     # (B, *example_shape)
        return x * self.std[...] + self.mean[...]

    def set_standardization(self, mean, std) -> None:
        if not self._standardize:
            raise ValueError(
                "TransformerFlow built with standardize=False")
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)


def make_tarflow(rngs, dim=None, cond_dim=0, *, modeled="vector",
                 img_size=None, patch_size=None, img_channels=1,
                 cond="add", cond_img_size=None, cond_patch_size=None,
                 cond_channels=1, prefix_tokens=1,
                 channels=64, num_blocks=8, layers_per_block=2, head_dim=16,
                 block_size=1, permutation="flip", standardize=True,
                 zero_init=True, use_softplus=True, soft_clip=4.0):
    """Build a TransformerFlow stack. ``modeled`` selects the tokenizer
    (vector/image); ``cond`` selects the conditioner (additive bias /
    vector-prefix / image-prefix). Vector defaults reproduce v1."""
    if modeled == "vector":
        tokenizer = VectorTokenizer(dim, block_size)
    elif modeled == "image":
        tokenizer = ImageTokenizer(img_size, img_size, img_channels, patch_size)
    else:
        raise ValueError(f"unknown modeled {modeled!r}")
    T, F = tokenizer.T, tokenizer.F

    def make_cond():
        if cond == "add":
            return VectorConditioner(cond_dim, channels, rngs=rngs)
        if cond == "vector_prefix":
            return VectorPrefixConditioner(cond_dim, channels, prefix_tokens,
                                           rngs=rngs)
        if cond == "image_prefix":
            m = (cond_img_size // cond_patch_size) ** 2
            return ImagePrefixConditioner(cond_channels, cond_patch_size,
                                          channels, m, rngs=rngs)
        raise ValueError(f"unknown cond {cond!r}")

    blocks = []
    for i in range(num_blocks):
        if permutation == "flip":
            perm = jnp.arange(T) if i % 2 == 0 else jnp.arange(T)[::-1]
        elif permutation == "random":
            perm = jax.random.permutation(rngs.params(), T)
        else:
            raise ValueError(f"unknown permutation {permutation!r}")
        blocks.append(MetaBlock(
            F=F, channels=channels, T=T, perm=perm, inv_perm=jnp.argsort(perm),
            conditioner=make_cond(), num_layers=layers_per_block,
            head_dim=head_dim, expansion=4, rngs=rngs, zero_init=zero_init,
            use_softplus=use_softplus, soft_clip=soft_clip))
    return TransformerFlow(blocks, tokenizer, dim, cond_dim,
                           standardize=standardize)
