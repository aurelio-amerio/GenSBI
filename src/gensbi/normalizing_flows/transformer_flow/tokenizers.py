"""Invertible tokenizers for the transformer flow (the modeled-variable seam).

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

A tokenizer maps the modeled variable to a token sequence ``(B, T, F)`` and back.
It MUST be volume-preserving (a fixed invertible reshape, log-det 0) — never a
learned lossy encoder — so the change-of-variables stays exact.
"""

from jax import Array


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

    def tokenize(self, x: Array) -> Array:
        return x.reshape(x.shape[0], self.T, self.F)

    def detokenize(self, tokens: Array) -> Array:
        return tokens.reshape(tokens.shape[0], self.dim)
