"""Bijection abstract base and the non-trainable Mask variable type."""

from abc import abstractmethod

from flax import nnx
from jax import Array


class Mask(nnx.Variable):
    """A fixed buffer (e.g. an autoregressive mask).

    Subclassing ``nnx.Variable`` (not ``nnx.Param``) keeps it out of
    ``nnx.split(wrt=nnx.Param)`` and the optimizer, while checkpointing still
    saves/restores it.
    """


class Bijection(nnx.Module):
    """Invertible map with the locked direction convention.

    Both methods act on a single example and return ``(output, log_det)`` where
    ``log_det`` is the log-abs-det of *that method's* Jacobian.

    - ``forward``:  noise -> data   (sampling; MAF: slow, sequential)
    - ``inverse``:  data  -> noise  (density; MAF: fast, one pass)
    """

    @abstractmethod
    def forward(self, u: Array, cond: Array | None = None) -> tuple[Array, Array]:
        ...  # pragma: no cover

    @abstractmethod
    def inverse(self, x: Array, cond: Array | None = None) -> tuple[Array, Array]:
        ...  # pragma: no cover
