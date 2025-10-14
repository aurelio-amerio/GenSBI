"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline
from .flux1joint import Flux1JointFlowPipeline, Flux1JointDiffusionPipeline
from .flux1 import Flux1FlowPipeline, Flux1DiffusionPipeline

__all__ = [
    "SimformerFlowPipeline",
    "SimformerDiffusionPipeline",
    "Flux1JointFlowPipeline",
    "Flux1JointDiffusionPipeline",
    "Flux1FlowPipeline",
    "Flux1DiffusionPipeline",
]

# 97% coverage, need to improve pipeline to hit some branches
