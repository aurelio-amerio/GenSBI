"""
Cookie cutter modules for creating and training SBI models.
"""

from .conditional_pipeline import ConditionalDiffusionPipeline, ConditionalFlowPipeline
from .flux1 import Flux1DiffusionPipeline, Flux1FlowPipeline
from .flux1joint import Flux1JointDiffusionPipeline, Flux1JointFlowPipeline
from .joint_pipeline import JointDiffusionPipeline, JointFlowPipeline
from .simformer import SimformerDiffusionPipeline, SimformerFlowPipeline
from .unconditional_pipeline import (
    UnconditionalDiffusionPipeline,
    UnconditionalFlowPipeline,
)

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
    "UnconditionalFlowPipeline",
    "UnconditionalDiffusionPipeline",
    
    "VAE1DPipeline",
    "VAE2DPipeline",
]

# 97% coverage, need to improve pipeline to hit some branches
