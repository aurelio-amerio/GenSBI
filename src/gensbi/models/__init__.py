from .flux1 import FluxParams, Flux, FluxCFMLoss, FluxWrapper, FluxDiffLoss
from .simformer import (
    Simformer,
    SimformerParams,
    SimformerWrapper,
    SimformerCFMLoss,
    SimformerDiffLoss,
)

from .simformer2 import (
    Simformer2,
    Simformer2Params,
    Simformer2Wrapper,
    Simformer2CFMLoss,
    Simformer2DiffLoss,
)   

__all__ = [
    "Flux",
    "FluxParams",
    "FluxCFMLoss",
    "FluxDiffLoss",
    "FluxWrapper",

    "Simformer",
    "SimformerParams",
    "SimformerCFMLoss",
    "SimformerDiffLoss",
    "SimformerWrapper",

    "Simformer2",
    "Simformer2Params",
    "Simformer2Wrapper",
    "Simformer2CFMLoss",
    "Simformer2DiffLoss",
]

# coverage 79% still need to work on this
