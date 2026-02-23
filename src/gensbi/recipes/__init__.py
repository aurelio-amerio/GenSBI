"""
Cookie cutter modules for creating and training SBI models.
"""

from .simformer import SimformerFlowPipeline, SimformerDiffusionPipeline, SimformerSMPipeline
from .flux1joint import Flux1JointFlowPipeline, Flux1JointDiffusionPipeline, Flux1JointSMPipeline
from .flux1 import Flux1FlowPipeline, Flux1DiffusionPipeline, Flux1SMPipeline

from .joint_pipeline import JointDiffusionPipeline, JointFlowPipeline, JointSMPipeline
from .conditional_pipeline import ConditionalFlowPipeline, ConditionalDiffusionPipeline, ConditionalSMPipeline
from .unconditional_pipeline import UnconditionalFlowPipeline, UnconditionalDiffusionPipeline, UnconditionalSMPipeline


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

    "JointDiffusionPipeline",
    "JointFlowPipeline",
    "JointSMPipeline",
    "ConditionalFlowPipeline",
    "ConditionalDiffusionPipeline",
    "ConditionalSMPipeline",
    "UnconditionalFlowPipeline",
    "UnconditionalDiffusionPipeline",
    "UnconditionalSMPipeline",

    "VAE1DPipeline",
    "VAE2DPipeline",
]

