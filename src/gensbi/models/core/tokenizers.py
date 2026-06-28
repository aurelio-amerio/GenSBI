"""Invertible tokenizers — the modeled-variable reshape seam.

Adapted from apple/ml-tarflow (TarFlow); see models/tarflow/LICENSE.apple.

A tokenizer maps the modeled variable to a token sequence ``(B, T, F)`` and back.
It MUST be volume-preserving (a fixed invertible reshape, log-det 0) — never a
learned lossy encoder — so the change-of-variables stays exact. Pure reshape, no
parameters, which is why it is a shared core primitive.
"""

from jax import Array
from gensbi.models.core.patching import patchify_2d, depatchify_2d


class VectorTokenizer:
    """Reshape a flat vector into a token sequence.

    Maps ``(B, dim)`` vectors to ``(B, T, F)`` token sequences via a
    volume-preserving reshape (log-det 0, no learned parameters). The number
    of tokens is ``T = dim // block_size`` and each token has ``F = block_size``
    features.

    Parameters
    ----------
    dim : int
        Total feature dimension of the input vector.
    block_size : int, optional
        Number of features per token.  Must divide ``dim``.  Default is 1.
    channels : int, optional
        Number of channels in the input.  Default is 1 (flat vector
        ``(B, dim)``).  When > 1 the input is ``(B, dim, C)`` and each
        token carries ``F = block_size * channels`` features.

    Raises
    ------
    ValueError
        If ``block_size`` does not divide ``dim``, or if ``channels < 1``.
    """

    def __init__(self, dim: int, block_size: int = 1, channels: int = 1):
        if dim % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide dim ({dim})")
        if channels < 1:
            raise ValueError(
                f"channels must be >= 1, got {channels}")
        self.dim = dim
        self.channels = channels
        self.F = block_size * channels
        self.T = dim // block_size
        self.example_shape = (dim,) if channels == 1 else (dim, channels)

    def tokenize(self, x: Array) -> Array:
        """Reshape a flat vector into a token sequence.

        Parameters
        ----------
        x : Array
            Input of shape ``(B, dim)`` when ``channels == 1``, or
            ``(B, dim, C)`` when ``channels > 1``.

        Returns
        -------
        Array
            Token sequence of shape ``(B, T, F)`` where
            ``T = dim // block_size`` and ``F = block_size * channels``.
        """
        return x.reshape(x.shape[0], self.T, self.F)

    def detokenize(self, tokens: Array) -> Array:
        """Flatten a token sequence back into a (channelled) vector.

        Parameters
        ----------
        tokens : Array
            Token sequence of shape ``(B, T, F)``.

        Returns
        -------
        Array
            Vector of shape ``(B, dim)`` when ``channels == 1``, or
            ``(B, dim, channels)`` when ``channels > 1``.
        """
        B = tokens.shape[0]
        if self.channels == 1:
            return tokens.reshape(B, self.dim)
        return tokens.reshape(B, self.dim, self.channels)


class ImageTokenizer:
    """Patchify a 2D image into a token sequence via :func:`patchify_2d`.

    Maps ``(B, H, W, C)`` images to ``(B, T, F)`` token sequences where
    ``T = (H // patch_size) * (W // patch_size)`` and
    ``F = C * patch_size * patch_size``. Pure reshape: volume-preserving
    (log-det 0, no learned parameters). Tokens are in raster (row-major)
    causal order as fixed by :func:`patchify_2d`.

    Parameters
    ----------
    height : int
        Image height in pixels. Must be divisible by ``patch_size``.
    width : int
        Image width in pixels. Must be divisible by ``patch_size``.
    channels : int
        Number of image channels.
    patch_size : int
        Patch edge length in pixels. Must divide both ``height`` and ``width``.

    Raises
    ------
    ValueError
        If ``patch_size`` does not divide ``height`` or ``width``.
    """

    def __init__(self, height: int, width: int, channels: int, patch_size: int):
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must divide height ({height}) and "
                f"width ({width})")
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.grid = (height // patch_size, width // patch_size)
        self.T = self.grid[0] * self.grid[1]
        self.F = channels * patch_size * patch_size
        self.example_shape = (height, width, channels)

    def tokenize(self, x: Array) -> Array:
        """Patchify an image into a token sequence.

        Parameters
        ----------
        x : Array
            Image of shape ``(B, H, W, C)``.

        Returns
        -------
        Array
            Token sequence of shape ``(B, T, F)`` where
            ``T = (H // patch_size) * (W // patch_size)`` and
            ``F = C * patch_size * patch_size``.
        """
        return patchify_2d(x, size=self.patch_size)

    def detokenize(self, tokens: Array) -> Array:
        """Reconstruct an image from a token sequence.

        Parameters
        ----------
        tokens : Array
            Token sequence of shape ``(B, T, F)``.

        Returns
        -------
        Array
            Image of shape ``(B, H, W, C)``.
        """
        return depatchify_2d(tokens, size=self.patch_size, grid=self.grid)
