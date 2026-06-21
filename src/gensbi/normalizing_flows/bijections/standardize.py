"""Fixed affine standardization bijection (non-trainable mean/std buffers)."""

import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Standardize(Bijection):
    """``inverse``: ``u = (x - mean) / std`` (data->standardized).

    ``forward``: ``x = u * std + mean``. log-det(inverse) ``= -sum(log std)``.
    Buffers are :class:`Mask` (non-Param); default to identity.
    """

    def __init__(self, dim: int):
        self.mean = Mask(jnp.zeros((dim,)))
        self.std = Mask(jnp.ones((dim,)))

    def set_stats(self, mean: Array, std: Array) -> None:
        self.mean.value = jnp.asarray(mean, dtype=self.mean.value.dtype)
        self.std.value = jnp.asarray(std, dtype=self.std.value.dtype)

    def inverse(self, x: Array, cond: Array | None = None):
        u = (x - self.mean.value) / self.std.value
        return u, -jnp.sum(jnp.log(self.std.value))

    def forward(self, u: Array, cond: Array | None = None):
        x = u * self.std.value + self.mean.value
        return x, jnp.sum(jnp.log(self.std.value))
