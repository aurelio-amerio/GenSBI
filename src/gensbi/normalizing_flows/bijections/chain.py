"""Compose bijections. Stored in noise->data (forward) order."""

import jax.numpy as jnp
from jax import Array
from flax import nnx

from gensbi.normalizing_flows.bijections.base import Bijection


class Chain(Bijection):
    """Sequential composition of bijections.

    Applies bijections in the order given for :meth:`forward` (noise→data)
    and in reversed order for :meth:`inverse` (data→noise).  Log-absolute
    determinants accumulate by summation.  ``bijections[-1]`` is closest to
    data space; it is the last bijection applied in :meth:`forward` and the
    first applied in :meth:`inverse`.

    Parameters
    ----------
    bijections : list of Bijection
        Ordered list of bijections stored in noise→data (forward) order.
    """

    def __init__(self, bijections: list[Bijection]):
        self.bijections = nnx.List(bijections)

    def forward(self, u: Array, cond: Array | None = None):
        """Map noise to data by composing all bijections in order.

        Parameters
        ----------
        u : Array
            Noise-space input.
        cond : Array or None, optional
            Conditioning input passed to every bijection, or ``None``.

        Returns
        -------
        x : Array
            Data-space output.
        logabsdet : Array
            Accumulated log absolute determinant of the composed forward map.
        """
        logdet = jnp.array(0.0)
        x = u
        for b in self.bijections:
            x, ld = b.forward(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: Array, cond: Array | None = None):
        """Map data to noise by composing all bijections in reversed order.

        Parameters
        ----------
        x : Array
            Data-space input.
        cond : Array or None, optional
            Conditioning input passed to every bijection, or ``None``.

        Returns
        -------
        u : Array
            Noise-space output.
        logabsdet : Array
            Accumulated log absolute determinant of the composed inverse map.
        """
        logdet = jnp.array(0.0)
        u = x
        for b in reversed(self.bijections):
            u, ld = b.inverse(u, cond)
            logdet = logdet + ld
        return u, logdet
