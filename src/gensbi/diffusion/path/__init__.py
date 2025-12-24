"""
Probability paths for diffusion models.

This module provides probability path implementations for diffusion models,
including the EDM path from the paper "Elucidating the Design Space of 
Diffusion-Based Generative Models" (Karras et al., 2022).
"""
from .edm_path import EDMPath

__all__ = [
    "EDMPath",
]
