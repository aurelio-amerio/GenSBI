"""Shared, model-agnostic primitives (the reuse home across architectures)."""

from gensbi.models.core.patching import patchify_2d, depatchify_2d

__all__ = ["patchify_2d", "depatchify_2d"]
