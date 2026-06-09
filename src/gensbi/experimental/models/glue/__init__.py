"""
Glue modules that compose pretrained encoders with SBI models.

These wrappers run a (frozen) encoder on the raw conditioning data and forward
the resulting latents into an SBI model such as :class:`gensbi.models.Flux1`.
"""

from .embedder import Embedded1DModel, Embedded2DModel

__all__ = [
    "Embedded1DModel",
    "Embedded2DModel",
]
