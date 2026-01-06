"""
Model architectures for GenSBI.

This package provides transformer-based models for simulation-based inference,
including Flux1, Simformer, and autoencoder architectures, along with their
associated loss functions and wrappers.
"""
from .flux1 import Flux1Params, Flux1

from .simformer import (
    Simformer,
    SimformerParams,
)

from .flux1joint import (
    Flux1Joint,
    Flux1JointParams,
)   


from .losses import JointCFMLoss, JointDiffLoss, ConditionalCFMLoss, ConditionalDiffLoss, UnconditionalCFMLoss, UnconditionalDiffLoss   

from .wrappers import JointWrapper, ConditionalWrapper, UnconditionalWrapper

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
