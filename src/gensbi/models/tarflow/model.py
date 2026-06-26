"""TarFlow: transformer autoregressive normalizing flow.

Adapted from apple/ml-tarflow (TarFlow) and apple/ml-starflow (STARFlow);
see models/tarflow/LICENSE.apple and LICENSE.starflow.

Self-contained ``(B, T, F)`` density model (absorbs the former
``TransformerFlow`` container and the ``make_tarflow`` factory). Head sizing
follows the Flux1 convention: specify ``head_dim`` and ``num_heads``; total
width ``channels = head_dim * num_heads`` is derived.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.models.core.tokenizers import VectorTokenizer, ImageTokenizer
from gensbi.models.tarflow.blocks import MetaBlock
from gensbi.models.tarflow.conditioners import (
    VectorConditioner, VectorPrefixConditioner, ImagePrefixConditioner,
)
from gensbi.normalizing_flows.bijections.base import Mask

_LOG2PI = jnp.log(2.0 * jnp.pi)


@dataclass
class TarFlowParams:
    """Architecture parameters for :class:`TarFlow`.

    ``modeled`` selects the tokenizer (vector/image); ``cond`` selects the
    conditioner (additive bias / vector-prefix / image-prefix). Head sizing is
    ``(head_dim, num_heads)`` with ``channels = head_dim * num_heads`` derived.
    """

    rngs: nnx.Rngs
    dim: int | None = None
    cond_dim: int = 0
    modeled: str = "vector"
    img_size: int | None = None
    patch_size: int | None = None
    img_channels: int = 1
    cond: str = "add"
    cond_img_size: int | None = None
    cond_patch_size: int | None = None
    cond_channels: int = 1
    prefix_tokens: int = 1
    head_dim: int = 16
    num_heads: int = 4
    num_blocks: int = 8
    layers_per_block: int = 2
    block_size: int = 1
    permutation: str = "flip"
    standardize: bool = True
    zero_init: bool = True
    use_softplus: bool = True
    soft_clip: float = 4.0

    def __post_init__(self):
        if self.modeled not in ("vector", "image"):
            raise ValueError(f"unknown modeled {self.modeled!r}")
        if self.modeled == "vector" and self.dim is None:
            raise ValueError("modeled='vector' requires dim")
        if self.modeled == "image" and (self.img_size is None or self.patch_size is None):
            raise ValueError("modeled='image' requires img_size and patch_size")
        if self.cond not in ("add", "vector_prefix", "image_prefix"):
            raise ValueError(f"unknown cond {self.cond!r}")
        if self.cond == "image_prefix" and (self.cond_img_size is None or self.cond_patch_size is None):
            raise ValueError("cond='image_prefix' requires cond_img_size and cond_patch_size")
        if self.permutation not in ("flip", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")
        self.channels = self.head_dim * self.num_heads


class TarFlow(nnx.Module):
    """Stack of MetaBlocks + tokenizer + standardization + N(0, I) base."""

    def __init__(self, params: TarFlowParams):
        rngs = params.rngs
        channels = params.channels

        if params.modeled == "vector":
            tokenizer = VectorTokenizer(params.dim, params.block_size)
        else:
            tokenizer = ImageTokenizer(params.img_size, params.img_size,
                                       params.img_channels, params.patch_size)
        T, F = tokenizer.T, tokenizer.F

        def make_cond():
            if params.cond == "add":
                return VectorConditioner(params.cond_dim, channels, rngs=rngs)
            if params.cond == "vector_prefix":
                return VectorPrefixConditioner(params.cond_dim, channels,
                                               params.prefix_tokens, rngs=rngs)
            m = (params.cond_img_size // params.cond_patch_size) ** 2
            return ImagePrefixConditioner(params.cond_channels,
                                          params.cond_patch_size, channels, m,
                                          rngs=rngs)

        blocks = []
        for i in range(params.num_blocks):
            if params.permutation == "flip":
                perm = jnp.arange(T) if i % 2 == 0 else jnp.arange(T)[::-1]
            else:
                perm = jax.random.permutation(rngs.params(), T)
            blocks.append(MetaBlock(
                F=F, channels=channels, T=T, perm=perm, inv_perm=jnp.argsort(perm),
                conditioner=make_cond(), num_layers=params.layers_per_block,
                num_heads=params.num_heads, expansion=4, rngs=rngs,
                zero_init=params.zero_init, use_softplus=params.use_softplus,
                soft_clip=params.soft_clip))

        self.blocks = nnx.List(blocks)
        self.tokenizer = tokenizer
        self.dim = params.dim
        self.cond_dim = params.cond_dim
        self.T = T
        self.F = F
        self.example_shape = tokenizer.example_shape
        self._standardize = params.standardize
        self.mean = Mask(jnp.zeros(self.example_shape))
        self.std = Mask(jnp.ones(self.example_shape))

    def _base_log_prob(self, z: Array) -> Array:
        return -0.5 * jnp.sum(z ** 2, axis=(1, 2)) - 0.5 * self.T * self.F * _LOG2PI

    def _ensure_batched(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == len(self.example_shape):
            x = x[None]
        return x

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        x = self._ensure_batched(x)
        u = (x - self.mean[...]) / self.std[...]
        logdet = -jnp.sum(jnp.log(self.std[...]))
        z = self.tokenizer.tokenize(u)
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
        x = self.tokenizer.detokenize(x)
        return x * self.std[...] + self.mean[...]

    def set_standardization(self, mean, std) -> None:
        if not self._standardize:
            raise ValueError("TarFlow built with standardize=False")
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)
