"""Bijection abstract base and the non-trainable Mask variable type."""

from abc import abstractmethod

from flax import nnx
from jax import Array


class Mask(nnx.Variable):
    """Fixed (non-trainable) buffer for bijection metadata.

    A non-:class:`~flax.nnx.Param` :class:`~flax.nnx.Variable` subclass so
    that optimizers and EMA utilities skip it while checkpointing still
    saves and restores it.  Typical use: autoregressive masks in
    :class:`~gensbi.models.maf.made.MaskedAutoregressive`.
    """


class Bijection(nnx.Module):
    """Invertible map between a noise variable and a data variable.

    Subclasses implement a differentiable bijection with a direction
    convention fixed across the library: :meth:`forward` maps noise to data
    (the sampling direction) and :meth:`inverse` maps data to noise (the
    density-evaluation direction). Both directions also return the log
    absolute determinant of the Jacobian of the transform they apply.
    """

    @abstractmethod
    def forward(self, u: Array, cond: Array | None = None) -> tuple[Array, Array]:
        """Map noise to data (the sampling direction).

        Parameters
        ----------
        u : Array
            Noise-space input.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional map.

        Returns
        -------
        x : Array
            Data-space output.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the forward map.

        Raises
        ------
        NotImplementedError
            This is an abstract method; subclasses must override it.
        """
        ...  # pragma: no cover

    @abstractmethod
    def inverse(self, x: Array, cond: Array | None = None) -> tuple[Array, Array]:
        """Map data to noise (the density-evaluation direction).

        Parameters
        ----------
        x : Array
            Data-space input.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional map.

        Returns
        -------
        u : Array
            Noise-space output.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the inverse map.

        Raises
        ------
        NotImplementedError
            This is an abstract method; subclasses must override it.
        """
        ...  # pragma: no cover
