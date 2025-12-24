"""
Utility functions for flow matching.

This module provides helper functions for tensor manipulation and array operations.
"""
from .utils import expand_tensor_like, unsqueeze_to_match

__all__ = [
    "unsqueeze_to_match",
    "expand_tensor_like",
]
