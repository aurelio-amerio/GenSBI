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
    """Embed a vector condition as a per-token additive bias.

    A two-layer MLP maps the condition to a ``channels``-dimensional vector
    that is broadcast-added to every token in the sequence. When
    ``cond_dim == 0`` the conditioner is unconditional and :meth:`embed`
    returns ``(None, None)``.

    Parameters
    ----------
    cond_dim : int
        Condition dimensionality. Set to ``0`` for an unconditional model.
    channels : int
        Output channel width matching the transformer embedding dimension.
    rngs : nnx.Rngs
        Flax RNG container for linear layer initialization.
    """

    def __init__(self, cond_dim: int, channels: int, rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond: Array | None):
        """Embed the condition into a per-token additive bias.

        Parameters
        ----------
        cond : Array or None
            Condition vector of shape ``(B, cond_dim)``, or ``None`` when the
            model is unconditional (``cond_dim == 0``).

        Returns
        -------
        bias : Array or None
            Per-token additive bias of shape ``(B, channels)``, or ``None``
            when ``cond_dim == 0``.
        prefix : None
            This conditioner does not produce prefix tokens; always ``None``.

        Raises
        ------
        ValueError
            If ``cond`` is ``None`` when ``cond_dim > 0``.
        """
        if self.cond_dim == 0:
            return (None, None)
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        bias = self.l2(jax.nn.silu(self.l1(cond)))
        return (bias, None)


class VectorPrefixConditioner(nnx.Module):
    """Embed a vector condition as prefix tokens prepended to the sequence.

    A linear projection maps the condition to ``num_tokens`` prefix tokens of
    width ``channels``, with learned positional embeddings added.

    Parameters
    ----------
    cond_dim : int
        Condition dimensionality.
    channels : int
        Output channel width matching the transformer embedding dimension.
    num_tokens : int
        Number of prefix tokens to produce (``M``).
    rngs : nnx.Rngs
        Flax RNG container for linear layer and positional embedding
        initialization.
    """

    def __init__(self, cond_dim: int, channels: int, num_tokens: int, rngs: nnx.Rngs):
        self.channels = channels
        self.M = num_tokens
        self.proj = nnx.Linear(cond_dim, channels * num_tokens, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, channels)) * 1e-2)

    def embed(self, cond: Array | None):
        """Embed the condition into prefix tokens.

        Parameters
        ----------
        cond : Array or None
            Condition vector of shape ``(B, cond_dim)``.

        Returns
        -------
        bias : None
            This conditioner does not produce a per-token additive bias;
            always ``None``.
        prefix : Array
            Prefix token sequence of shape ``(B, num_tokens, channels)`` with
            learned positional embeddings added.

        Raises
        ------
        ValueError
            If ``cond`` is ``None``.
        """
        if cond is None:
            raise ValueError("cond is required for VectorPrefixConditioner")
        B = cond.shape[0]
        h = self.proj(cond).reshape(B, self.M, self.channels)
        return (None, h + self.pos[...][None])


class ImagePrefixConditioner(nnx.Module):
    """Embed an image condition as prefix tokens prepended to the sequence.

    Patchifies a spatial image ``(B, H, W, C)`` into
    ``M = (H / patch_size) * (W / patch_size)`` flat patch vectors, projects
    each patch to ``channels`` dimensions, and adds learned positional
    embeddings.

    Parameters
    ----------
    cond_channels : int
        Number of channels in the conditioning image.
    patch_size : int
        Spatial size of each square patch (height and width in pixels).
    channels : int
        Output channel width matching the transformer embedding dimension.
    num_tokens : int
        Number of prefix tokens; must equal
        ``(H / patch_size) * (W / patch_size)``.
    rngs : nnx.Rngs
        Flax RNG container for projection layer and positional embedding
        initialization.
    """

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
        """Patchify an image condition and embed it as prefix tokens.

        Parameters
        ----------
        cond : Array or None
            Image condition of shape ``(B, H, W, C)``.

        Returns
        -------
        bias : None
            This conditioner does not produce a per-token additive bias;
            always ``None``.
        prefix : Array
            Prefix token sequence of shape ``(B, num_tokens, channels)`` with
            learned positional embeddings added.

        Raises
        ------
        ValueError
            If ``cond`` is ``None``.
        """
        if cond is None:
            raise ValueError("cond is required for ImagePrefixConditioner")
        patches = patchify_2d(cond, size=self.patch_size)      # (B, M, in_f)
        return (None, self.proj(patches) + self.pos[...][None])
