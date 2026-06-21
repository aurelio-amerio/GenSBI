"""Discrete (autoregressive) normalizing flows for GenSBI.

A parallel track to the flow-matching/diffusion methods: the flow IS the
density model, with exact ``log_prob`` and one-pass conditional ``sample``.
"""

from gensbi.normalizing_flows.flow import Flow, make_maf

__all__ = ["Flow", "make_maf"]
