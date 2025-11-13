"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline
from .flux1joint import Flux1JointFlowPipeline, Flux1JointDiffusionPipeline
from .flux1 import Flux1FlowPipeline, Flux1DiffusionPipeline

from .joint_pipeline import JointDiffusionPipeline, JointFlowPipeline
from .conditional_pipeline import ConditionalFlowPipeline, ConditionalDiffusionPipeline

__all__ = [
    "SimformerFlowPipeline",
    "SimformerDiffusionPipeline",
    "Flux1JointFlowPipeline",
    "Flux1JointDiffusionPipeline",
    "Flux1FlowPipeline",
    "Flux1DiffusionPipeline",
    
    "JointDiffusionPipeline",
    "JointFlowPipeline",
    "ConditionalFlowPipeline",
    "ConditionalDiffusionPipeline",
]

# 97% coverage, need to improve pipeline to hit some branches
