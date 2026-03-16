"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline, SimformerSMPipeline
from .flux1joint import Flux1JointFlowPipeline, Flux1JointDiffusionPipeline, Flux1JointSMPipeline
from .flux1 import Flux1FlowPipeline, Flux1DiffusionPipeline, Flux1SMPipeline

# Unified pipelines
from .conditional_pipeline import ConditionalPipeline
from .joint_pipeline import JointPipeline
from .unconditional_pipeline import UnconditionalPipeline


__all__ = [
    "SimformerFlowPipeline",
    "SimformerDiffusionPipeline",
    "SimformerSMPipeline",
    "Flux1JointFlowPipeline",
    "Flux1JointDiffusionPipeline",
    "Flux1JointSMPipeline",
    "Flux1FlowPipeline",
    "Flux1DiffusionPipeline",
    "Flux1SMPipeline",

    "ConditionalPipeline",
    "JointPipeline",
    "UnconditionalPipeline",

    "VAE1DPipeline",
    "VAE2DPipeline",
]
