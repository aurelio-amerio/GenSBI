"""Transformer autoregressive normalizing-flow density model.

Exports :class:`TarFlow` and its configuration dataclass
:class:`TarFlowParams` for building autoregressive normalizing flows
based on a stack of causal transformer blocks (MetaBlocks).
"""

from gensbi.models.tarflow.model import TarFlowParams, TarFlow

__all__ = ["TarFlowParams", "TarFlow"]
