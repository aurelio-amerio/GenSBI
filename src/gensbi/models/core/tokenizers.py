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
    """Reshape a vector ``(B, dim)`` into ``T = dim // block_size`` tokens of
    ``F = block_size`` features. Pure reshape: invertible, log-det 0."""

    def __init__(self, dim: int, block_size: int = 1):
        if dim % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide dim ({dim})")
        self.dim = dim
        self.F = block_size
        self.T = dim // block_size
        self.example_shape = (dim,)

    def tokenize(self, x: Array) -> Array:
        return x.reshape(x.shape[0], self.T, self.F)

    def detokenize(self, tokens: Array) -> Array:
        return tokens.reshape(tokens.shape[0], self.dim)


class ImageTokenizer:
    """Patchify an image ``(B, H, W, C)`` into ``T = (H/p)(W/p)`` tokens of
    ``F = C*p*p`` features via :func:`patchify_2d`. Pure reshape: invertible,
    log-det 0. Raster causal order (fixed by ``patchify_2d``)."""

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
        return patchify_2d(x, size=self.patch_size)

    def detokenize(self, tokens: Array) -> Array:
        return depatchify_2d(tokens, size=self.patch_size, grid=self.grid)
