"""
Probability paths for flow matching.

This module provides probability path implementations for flow matching algorithms,
including affine paths and conditional optimal transport paths.
"""
# FIXME: some features not yet implemented as they are not used for sbi

from .affine import AffineProbPath, CondOTProbPath
from .path import ProbPath
from .path_sample import PathSample


__all__ = [
    "ProbPath",
    "PathSample",
    "AffineProbPath",
    "CondOTProbPath",
]
