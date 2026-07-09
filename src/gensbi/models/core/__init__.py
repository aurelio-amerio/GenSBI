"""Shared, model-agnostic primitives (the reuse home across architectures)."""

from gensbi.models.core.patching import patchify_2d, depatchify_2d
from gensbi.models.core.tokenizers import VectorTokenizer, ImageTokenizer

__all__ = ["patchify_2d", "depatchify_2d", "VectorTokenizer", "ImageTokenizer"]
