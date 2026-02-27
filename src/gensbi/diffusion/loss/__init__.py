"""
Diffusion loss functions with unified interface.

This subpackage provides :class:`EDMLoss` and :class:`SMLoss`, which wrap
the path-specific loss functions into a uniform
``(model, batch, condition_mask, model_extras)`` interface.
"""

from .edm_loss import EDMLoss
from .sm_loss import SMLoss

__all__ = [
    "EDMLoss",
    "SMLoss",
]
