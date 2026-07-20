"""Pluggable condition embedders for FieldDiT (Phase 1: scalar / vector).

A condition is embedded into a token stream (consumed by the MMDiT core via
joint attention) plus a pooled summary (added to the modulation vector for
flagged-C decoder modulation).
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike


class ScalarCondEmbedder(nnx.Module):
    """Embed a few condition tokens (e.g. theta scalars) to ``hidden_size``.

    Input ``cond`` is ``(B, k, in_channels)`` (or ``(B, k)``, auto-expanded to
    ``(B, k, 1)`` — the 2D form raises unless ``in_channels == 1``).
    Returns ``(cond_tokens (B, k, hidden), summary (B, hidden))`` where the
    summary is a projection of the mean-pooled tokens.
    """

    def __init__(
        self, in_channels: int, hidden_size: int, rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.bfloat16, param_dtype: DTypeLike = jnp.float32,
    ):
        self.in_channels = in_channels
        self.token_proj = nnx.Linear(
            in_features=in_channels, out_features=hidden_size, use_bias=True,
            rngs=rngs, dtype=dtype, param_dtype=param_dtype,
        )
        self.summary_proj = nnx.Linear(
            in_features=hidden_size, out_features=hidden_size, use_bias=True,
            rngs=rngs, dtype=dtype, param_dtype=param_dtype,
        )

    def __call__(self, cond):
        if cond.ndim == 2:
            if self.in_channels != 1:
                raise ValueError(
                    f"2D cond (B, k) is only valid when cond_in_channels == 1; "
                    f"this embedder has cond_in_channels={self.in_channels} — "
                    f"pass cond as (B, k, {self.in_channels})"
                )
            cond = cond[..., None]
        tokens = self.token_proj(cond)
        summary = self.summary_proj(jnp.mean(tokens, axis=1))
        return tokens, summary
