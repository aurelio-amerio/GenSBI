"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline
from .flux import FluxFlowPipeline, FluxDiffusionPipeline

__all__ = [
    "SimformerFlowPipeline",
    "SimformerDiffusionPipeline",
    "FluxFlowPipeline",
    "FluxDiffusionPipeline",
]

# 97% coverage, need to improve pipeline to hit some branches
