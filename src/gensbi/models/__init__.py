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

from .wrappers import JointWrapper, ConditionalWrapper, UnconditionalWrapper

from .maf import MAFlowParams, MAFlow

from .tarflow import TarFlowParams, TarFlow

from .healswin import HealSwinEncoder, HealSwinParams

__all__ = [
    "Flux1",
    "Flux1Params",
    "Simformer",
    "SimformerParams",
    "Flux1Joint",
    "Flux1JointParams",
    "JointWrapper",
    "ConditionalWrapper",
    "UnconditionalWrapper",
    "MAFlowParams",
    "MAFlow",
    "TarFlowParams",
    "TarFlow",
    "HealSwinEncoder",
    "HealSwinParams",
]
