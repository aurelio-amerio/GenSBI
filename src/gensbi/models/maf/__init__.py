"""Masked Autoregressive Flow (MAF) density model.

Provides :class:`MAFlow`, a normalizing flow for exact log-density evaluation
and sampling built from stacked autoregressive layers, and its configuration
dataclass :class:`MAFlowParams`.
"""

from gensbi.models.maf.model import MAFlowParams, MAFlow

__all__ = ["MAFlowParams", "MAFlow"]
