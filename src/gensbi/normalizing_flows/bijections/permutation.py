"""Permutation bijection (dimension reordering between flow layers)."""

import jax
import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Permutation(Bijection):
    """Reorder dims; ``cond`` is ignored; log-det is 0.

    ``perm`` and its inverse are stored as :class:`Mask` buffers (non-Param).
    """

    def __init__(self, perm: Array):
        perm = jnp.asarray(perm, dtype=jnp.int32)
        self.perm = Mask(perm)
        self.inv_perm = Mask(jnp.argsort(perm))

    @classmethod
    def reverse(cls, dim: int) -> "Permutation":
        return cls(jnp.arange(dim)[::-1])

    @classmethod
    def random(cls, dim: int, rngs) -> "Permutation":
        return cls(jax.random.permutation(rngs.params(), dim))

    def inverse(self, x: Array, cond: Array | None = None):
        return x[self.perm.value], jnp.array(0.0)

    def forward(self, u: Array, cond: Array | None = None):
        return u[self.inv_perm.value], jnp.array(0.0)
