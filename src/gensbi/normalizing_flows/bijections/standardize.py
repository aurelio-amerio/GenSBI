"""Fixed affine standardization bijection (non-trainable mean/std buffers)."""

import jax.numpy as jnp
from jax import Array

from gensbi.normalizing_flows.bijections.base import Bijection, Mask


class Standardize(Bijection):
    """Fixed affine standardization using non-trainable mean and std buffers.

    Buffers default to identity (mean 0, std 1) and can be updated in place
    via :meth:`set_stats`.  They are stored as
    :class:`~gensbi.normalizing_flows.bijections.base.Mask` variables so
    that optimizers and EMA utilities skip them.

    Parameters
    ----------
    dim : int
        Dimension of the data vector (length of mean and std buffers).
    """

    def __init__(self, dim: int):
        self.mean = Mask(jnp.zeros((dim,)))
        self.std = Mask(jnp.ones((dim,)))

    def set_stats(self, mean: Array, std: Array) -> None:
        """Update the mean and standard-deviation buffers in place.

        Parameters
        ----------
        mean : Array
            New mean values of shape ``(dim,)``.
        std : Array
            New standard-deviation values of shape ``(dim,)``; must be
            strictly positive.

        Returns
        -------
        None
            This method modifies the buffers in place and returns nothing.
        """
        self.mean[...] = jnp.asarray(mean, dtype=self.mean[...].dtype)
        self.std[...] = jnp.asarray(std, dtype=self.std[...].dtype)

    def inverse(self, x: Array, cond: Array | None = None):
        """Map data to noise by standardizing: ``u = (x - mean) / std``.

        Parameters
        ----------
        x : Array
            Data-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Ignored; present for interface compatibility.

        Returns
        -------
        u : Array
            Standardized noise-space output.
        logabsdet : Array
            Log absolute determinant of the inverse map: ``-sum(log std)``.
        """
        u = (x - self.mean[...]) / self.std[...]
        return u, -jnp.sum(jnp.log(self.std[...]))

    def forward(self, u: Array, cond: Array | None = None):
        """Map noise to data by destandardizing: ``x = u * std + mean``.

        Parameters
        ----------
        u : Array
            Noise-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Ignored; present for interface compatibility.

        Returns
        -------
        x : Array
            Destandardized data-space output.
        logabsdet : Array
            Log absolute determinant of the forward map: ``sum(log std)``.
        """
        x = u * self.std[...] + self.mean[...]
        return x, jnp.sum(jnp.log(self.std[...]))
