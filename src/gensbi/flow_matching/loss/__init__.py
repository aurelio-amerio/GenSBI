"""
Loss functions for flow matching.

This module provides loss functions for training continuous flow matching models.
"""
from .continuous_loss import ContinuousFMLoss


__all__ = [
    "ContinuousFMLoss",
]
