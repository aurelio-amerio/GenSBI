"""Compose bijections. Stored in noise->data (forward) order."""

import jax.numpy as jnp
from jax import Array
from flax import nnx

from gensbi.normalizing_flows.bijections.base import Bijection


class Chain(Bijection):
    """Apply bijections in order for ``forward``, reversed for ``inverse``.

    Log-dets accumulate (sum). ``bijections[-1]`` is closest to data; it is the
    first applied in ``inverse`` and last in ``forward``.
    """

    def __init__(self, bijections: list[Bijection]):
        self.bijections = nnx.List(bijections)

    def forward(self, u: Array, cond: Array | None = None):
        logdet = jnp.array(0.0)
        x = u
        for b in self.bijections:
            x, ld = b.forward(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: Array, cond: Array | None = None):
        logdet = jnp.array(0.0)
        u = x
        for b in reversed(self.bijections):
            u, ld = b.inverse(u, cond)
            logdet = logdet + ld
        return u, logdet
