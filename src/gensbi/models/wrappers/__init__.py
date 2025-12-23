"""
Model wrappers for flow matching.

This module provides wrapper classes for different model types (conditional, joint,
unconditional) used in flow matching and simulation-based inference.
"""
from .conditional import ConditionalWrapper
from .joint import JointWrapper
from .unconditional import UnconditionalWrapper

__all__ = [
    "ConditionalWrapper",
    "JointWrapper",
    "UnconditionalWrapper",
]