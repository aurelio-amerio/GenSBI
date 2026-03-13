"""
Core module for GenSBI.

Provides the ``GenerativeMethod`` strategy pattern and its concrete
implementations for flow matching, EDM diffusion, and score matching.

These strategy objects encapsulate the generative framework (path, loss,
solver, batch preparation) and are composed into mode-specific pipelines
(Conditional, Joint, Unconditional) in the ``recipes`` module.
"""

from gensbi.core.generative_method import GenerativeMethod
from gensbi.core.flow_matching import FlowMatchingMethod
from gensbi.core.diffusion_edm import DiffusionEDMMethod
from gensbi.core.score_matching import ScoreMatchingMethod
from gensbi.core.ode_solver import NewODESolver
from gensbi.core.sde_solver import NewSDESolver

__all__ = [
    "GenerativeMethod",
    "FlowMatchingMethod",
    "DiffusionEDMMethod",
    "ScoreMatchingMethod",
    "NewODESolver",
    "NewSDESolver",
]
