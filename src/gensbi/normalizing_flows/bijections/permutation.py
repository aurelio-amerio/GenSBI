"""Permutation bijection (dimension reordering between flow layers)."""

import jax
import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Permutation(Bijection):
    """Reorder dimensions; conditioning is ignored; log-det is always 0.

    Both the permutation index array and its inverse are stored as
    :class:`~gensbi.normalizing_flows.bijections.base.Mask` buffers so that
    optimizers and EMA utilities skip them.

    Parameters
    ----------
    perm : Array
        Integer index array of shape ``(dim,)`` that defines the reordering.
        ``perm[i]`` is the source index for output position ``i``.
    """

    def __init__(self, perm: Array):
        perm = jnp.asarray(perm, dtype=jnp.int32)
        self.perm = Mask(perm)
        self.inv_perm = Mask(jnp.argsort(perm))

    @classmethod
    def reverse(cls, dim: int) -> "Permutation":
        """Construct a permutation that reverses dimension order.

        Parameters
        ----------
        dim : int
            Number of dimensions.

        Returns
        -------
        Permutation
            A :class:`Permutation` whose :meth:`inverse` reverses the input.
        """
        return cls(jnp.arange(dim)[::-1])

    @classmethod
    def random(cls, dim: int, rngs) -> "Permutation":
        """Construct a uniformly random permutation.

        Parameters
        ----------
        dim : int
            Number of dimensions.
        rngs : nnx.Rngs
            Flax RNG container used to draw the random permutation index.

        Returns
        -------
        Permutation
            A :class:`Permutation` with a randomly shuffled index array.
        """
        return cls(jax.random.permutation(rngs.params(), dim))

    def inverse(self, x: Array, cond: Array | None = None):
        """Map data to noise by applying ``perm`` to reorder dimensions.

        Parameters
        ----------
        x : Array
            Data-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Ignored; present for interface compatibility.

        Returns
        -------
        u : Array
            Reordered noise-space output.
        logabsdet : Array
            Zero scalar (permutations have unit Jacobian determinant).
        """
        return x[self.perm[...]], jnp.array(0.0)

    def forward(self, u: Array, cond: Array | None = None):
        """Map noise to data by applying the inverse permutation.

        Parameters
        ----------
        u : Array
            Noise-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Ignored; present for interface compatibility.

        Returns
        -------
        x : Array
            Reordered data-space output.
        logabsdet : Array
            Zero scalar (permutations have unit Jacobian determinant).
        """
        return u[self.inv_perm[...]], jnp.array(0.0)
