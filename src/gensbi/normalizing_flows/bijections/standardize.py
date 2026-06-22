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
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)

    def inverse(self, x: Array, cond: Array | None = None):
        u = (x - self.mean[...]) / self.std[...]
        return u, -jnp.sum(jnp.log(self.std[...]))

    def forward(self, u: Array, cond: Array | None = None):
        x = u * self.std[...] + self.mean[...]
        return x, jnp.sum(jnp.log(self.std[...]))
