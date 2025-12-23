"""
Probability paths for diffusion models.

This module provides probability path implementations for diffusion models,
including the EDM (Elucidating Diffusion Models) path.
"""
from .edm_path import EDMPath

__all__ = [
    "EDMPath",
]
