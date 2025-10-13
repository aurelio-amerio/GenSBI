"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline
from .simformer2 import Simformer2FlowPipeline, Simformer2DiffusionPipeline
from .flux import FluxFlowPipeline, FluxDiffusionPipeline

__all__ = [
    "SimformerFlowPipeline",
    "SimformerDiffusionPipeline",
    "Simformer2FlowPipeline",
    "Simformer2DiffusionPipeline",
    "FluxFlowPipeline",
    "FluxDiffusionPipeline",
]

# 97% coverage, need to improve pipeline to hit some branches
