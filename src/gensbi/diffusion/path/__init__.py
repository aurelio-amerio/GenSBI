"""
Probability paths for diffusion models.

This module provides probability path implementations for diffusion models,
including the EDM path from the paper "Elucidating the Design Space of
Diffusion-Based Generative Models" (Karras et al., 2022) and the standard
score matching path from "Score-Based Generative Modeling through Stochastic
Differential Equations" (Song et al., 2021).
"""

from .edm_path import EDMPath
from .sm_path import SMPath

__all__ = [
    "EDMPath",
    "SMPath",
]
