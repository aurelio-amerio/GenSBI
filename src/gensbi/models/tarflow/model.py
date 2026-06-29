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
    AdditiveBiasConditioner, VectorConditioner, ImageConditioner,
)
from gensbi.normalizing_flows.bijections.base import Mask

_LOG2PI = jnp.log(2.0 * jnp.pi)


@dataclass
class TarFlowParams:
    """Architecture parameters for :class:`TarFlow`.

    ``modeled`` selects the tokenizer (``"vector"`` or ``"image"``); ``cond``
    selects the conditioner (``"bias"``, ``"vector"``, or
    ``"image"``). Head sizing follows the Flux1 convention: specify
    ``head_dim`` and ``num_heads``; total width
    ``channels = head_dim * num_heads`` is derived in ``__post_init__``.

    Parameters
    ----------
    rngs : nnx.Rngs
        Flax RNG container passed to all sub-modules during construction.
    dim : int or None, optional
        Feature dimension of each input vector. Required when
        ``modeled="vector"``. Default is ``None``.
    cond_dim : int, optional
        Dimensionality of the conditioning vector. Set to ``0`` for an
        unconditional model. Default is ``0``.
    modeled : str, optional
        Tokenizer type: ``"vector"`` (1-D data) or ``"image"`` (spatial data).
        Default is ``"vector"``.
    img_size : int or None, optional
        Spatial size (height = width) of the modeled image. Required when
        ``modeled="image"``. Default is ``None``.
    patch_size : int or None, optional
        Patch size for the image tokenizer. Required when
        ``modeled="image"``. Default is ``None``.
    img_channels : int, optional
        Number of channels in the modeled image. Default is ``1``.
    cond : str, optional
        Conditioning strategy: ``"bias"`` (per-token additive bias via
        :class:`~gensbi.models.tarflow.conditioners.AdditiveBiasConditioner`),
        ``"vector"`` (prefix tokens from a vector via
        :class:`~gensbi.models.tarflow.conditioners.VectorConditioner`),
        or ``"image"`` (prefix tokens from an image via
        :class:`~gensbi.models.tarflow.conditioners.ImageConditioner`).
        Default is ``"bias"``.
    cond_img_size : int or None, optional
        Spatial size of the conditioning image. Required when
        ``cond="image"``. Default is ``None``.
    cond_patch_size : int or None, optional
        Patch size for the image conditioning tokenizer. Required when
        ``cond="image"``. Default is ``None``.
    cond_channels : int, optional
        Number of channels in the conditioning image. Default is ``1``.
    prefix_tokens : int, optional
        Number of prefix tokens produced by ``cond="vector"``.
        Default is ``1``.
    head_dim : int, optional
        Dimension per attention head. Default is ``16``.
    num_heads : int, optional
        Number of attention heads per block. Default is ``4``.
    num_blocks : int, optional
        Number of :class:`~gensbi.models.tarflow.blocks.MetaBlock` layers.
        Default is ``8``.
    layers_per_block : int, optional
        Number of :class:`~gensbi.models.tarflow.blocks.AttentionBlock`
        layers inside each :class:`~gensbi.models.tarflow.blocks.MetaBlock`.
        Default is ``2``.
    block_size : int, optional
        Token grouping factor for the vector tokenizer. Default is ``1``.
    permutation : str, optional
        Token permutation strategy per block: ``"flip"`` (alternate
        forward/reverse order) or ``"random"`` (independently sampled per
        block). Default is ``"flip"``.
    standardize : bool, optional
        If ``True`` (default), apply mean/std standardization to inputs and
        outputs. Enables :meth:`TarFlow.set_standardization`.
    zero_init : bool, optional
        If ``True`` (default), initialize ``proj_out`` weights to zero so
        each :class:`~gensbi.models.tarflow.blocks.MetaBlock` starts as the
        identity map.
    use_softplus : bool, optional
        If ``True`` (default), use softplus for the affine scale (numerically
        stable, bounded tail). If ``False``, use ``exp`` (legacy behavior).
    soft_clip : float, optional
        Soft-clip magnitude applied via ``tanh`` to raw network outputs before
        splitting into ``(a, b)``. Default is ``4.0``.
    """

    rngs: nnx.Rngs
    dim: int | None = None
    cond_dim: int = 0
    modeled: str = "vector"
    img_size: int | None = None
    patch_size: int | None = None
    img_channels: int = 1
    vec_channels: int = 1
    cond: str = "bias"
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
        if self.cond not in ("bias", "vector", "image"):
            raise ValueError(f"unknown cond {self.cond!r}")
        if self.cond == "image" and (self.cond_img_size is None or self.cond_patch_size is None):
            raise ValueError("cond='image' requires cond_img_size and cond_patch_size")
        if self.permutation not in ("flip", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")
        self.channels = self.head_dim * self.num_heads


class TarFlow(nnx.Module):
    """Transformer autoregressive normalizing flow density model.

    Stacks :class:`~gensbi.models.tarflow.blocks.MetaBlock` bijections with
    alternating token permutations on top of a tokenizer and an isotropic
    Gaussian base distribution. Supports both vector and image data, with
    optional input standardization.

    Parameters
    ----------
    params : TarFlowParams
        Architecture and initialization parameters.
    """

    def __init__(self, params: TarFlowParams):
        rngs = params.rngs
        channels = params.channels

        if params.modeled == "vector":
            tokenizer = VectorTokenizer(params.dim, params.block_size,
                                        params.vec_channels)
        else:
            tokenizer = ImageTokenizer(params.img_size, params.img_size,
                                       params.img_channels, params.patch_size)
        T, F = tokenizer.T, tokenizer.F

        def make_cond():
            if params.cond == "bias":
                return AdditiveBiasConditioner(params.cond_dim, channels, rngs=rngs,
                                               cond_channels=params.cond_channels)
            if params.cond == "vector":
                return VectorConditioner(params.cond_dim, channels,
                                         params.prefix_tokens, rngs=rngs)
            m = (params.cond_img_size // params.cond_patch_size) ** 2
            return ImageConditioner(params.cond_channels,
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
        """Compute the log-probability of data under the model.

        Applies standardization, tokenizes the input, then runs each
        :class:`~gensbi.models.tarflow.blocks.MetaBlock`'s
        :meth:`~gensbi.models.tarflow.blocks.MetaBlock.inverse` transform
        (data→noise direction), accumulating the log-absolute-determinant
        terms, and finally evaluates the base Gaussian log-probability.

        Parameters
        ----------
        x : Array
            Data samples of shape ``(B, *example_shape)`` or a single
            unbatched sample that will be promoted to a batch of one.
        cond : Array or None, optional
            Conditioning input of shape ``(B, cond_dim)``, or ``None`` for
            an unconditional model.

        Returns
        -------
        Array
            Log-probabilities of shape ``(B,)``.
        """
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
        """Draw samples from the model.

        Samples noise from ``N(0, I)``, then applies each
        :class:`~gensbi.models.tarflow.blocks.MetaBlock`'s
        :meth:`~gensbi.models.tarflow.blocks.MetaBlock.forward` transform
        (noise→data direction) in reverse block order, detokenizes the
        result, and applies the inverse standardization.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for noise sampling.
        cond : Array or None, optional
            Conditioning input of shape ``(B, cond_dim)``. If provided,
            ``nsamples`` is inferred from ``cond.shape[0]``.
        nsamples : int or None, optional
            Number of samples to draw. Required when ``cond`` is ``None``.

        Returns
        -------
        Array
            Samples of shape ``(B, *example_shape)``.
        """
        if cond is not None:
            nsamples = cond.shape[0]
        z = jax.random.normal(key, (nsamples, self.T, self.F))
        x = z
        for blk in reversed(self.blocks):
            x, _ = blk.forward(x, cond)
        x = self.tokenizer.detokenize(x)
        return x * self.std[...] + self.mean[...]

    def _fit_stat(self, s, dtype):
        s = jnp.asarray(s, dtype=dtype)
        es = self.example_shape
        if s.ndim == 1 and s.shape[0] == es[0]:
            s = s.reshape((es[0],) + (1,) * (len(es) - 1))   # (dim,) -> (dim,1,...)
        return jnp.broadcast_to(s, es)

    def set_standardization(self, mean, std) -> None:
        """Set the mean and standard deviation for input standardization.

        Accepts shapes ``(dim,)`` (broadcast to ``(dim, 1)``), ``(dim, 1)``,
        ``(C,)`` (per-channel broadcast), or a scalar broadcastable to
        ``example_shape``.

        Parameters
        ----------
        mean : Array
            Mean broadcastable to ``example_shape``.
        std : Array
            Standard deviation broadcastable to ``example_shape``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model was built with ``standardize=False``.
        """
        if not self._standardize:
            raise ValueError("TarFlow built with standardize=False")
        self.mean[...] = self._fit_stat(mean, self.mean[...].dtype)
        self.std[...] = self._fit_stat(std, self.std[...].dtype)
