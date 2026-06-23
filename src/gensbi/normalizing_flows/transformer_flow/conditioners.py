"""Conditioning seams for the transformer flow.

Adapted from apple/ml-tarflow (TarFlow); see transformer_flow/LICENSE.apple.

v1 ``VectorConditioner`` is the continuous analog of TarFlow's ``class_embed``:
an MLP embeds the condition to a ``channels``-vector that is broadcast-added to
every token. The signal depends only on the condition (constant w.r.t. the
modeled variable), so it shifts the affine params without breaking the
triangular Jacobian. A plain 2-layer MLP is used (not ``MLPEmbedder``, whose
``hidden_dim % in_dim == 0`` constraint does not fit arbitrary ``cond_dim``).
"""

import jax
from flax import nnx
from jax import Array


class VectorConditioner(nnx.Module):
    """MLP(cond) → per-token additive bias. ``cond_dim == 0`` ⇒ unconditional."""

    def __init__(self, cond_dim: int, channels: int, rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond: Array | None) -> Array | None:
        if self.cond_dim == 0:
            return None
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        return self.l2(jax.nn.silu(self.l1(cond)))

    def inject(self, tokens: Array, signal: Array | None) -> Array:
        if signal is None:
            return tokens
        return tokens + signal[:, None, :]
