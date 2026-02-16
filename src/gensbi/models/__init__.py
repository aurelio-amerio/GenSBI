"""
Model architectures for GenSBI.

This package provides transformer-based models for simulation-based inference,
including Flux1, Simformer, and autoencoder architectures, along with their
associated loss functions and wrappers.
"""
from .flux1 import Flux1, Flux1Params
from .flux1joint import (
    Flux1Joint,
    Flux1JointParams,
)
from .losses import (
    ConditionalCFMLoss,
    ConditionalDiffLoss,
    JointCFMLoss,
    JointDiffLoss,
    UnconditionalCFMLoss,
    UnconditionalDiffLoss,
)
from .simformer import (
    Simformer,
    SimformerParams,
)
from .wrappers import ConditionalWrapper, JointWrapper, UnconditionalWrapper

__all__ = [
    "Flux1",
    "Flux1Params",

    "Simformer",
    "SimformerParams",

    "Flux1Joint",
    "Flux1JointParams",

    "JointCFMLoss",
    "JointDiffLoss",
    "ConditionalCFMLoss",
    "ConditionalDiffLoss",
    "UnconditionalCFMLoss",
    "UnconditionalDiffLoss",
    
    "JointWrapper",
    "ConditionalWrapper",
    "UnconditionalWrapper",
]

# coverage 79% still need to work on this
