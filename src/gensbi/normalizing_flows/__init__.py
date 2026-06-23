"""Discrete (autoregressive) normalizing flows for GenSBI.

A parallel track to the flow-matching/diffusion methods: the flow IS the
density model, with exact ``log_prob`` and one-pass conditional ``sample``.
"""

from gensbi.normalizing_flows.flow import Flow, make_maf
from gensbi.normalizing_flows.bijections.transformers import Affine, RQSpline
from gensbi.normalizing_flows.transformer_flow import TransformerFlow, make_tarflow

__all__ = ["Flow", "make_maf", "Affine", "RQSpline", "TransformerFlow", "make_tarflow"]
