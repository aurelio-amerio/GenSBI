"""The Flow module: base distribution + Chain of bijections."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine


class Flow(nnx.Module):
    """Normalizing flow over ``(batch, dim)`` data, optionally conditioned.

    ``log_prob(x, cond) = base.log_prob(u) + logdet`` with
    ``u, logdet = chain.inverse(x, cond)``. The base is a standard normal over
    ``(dim,)``, built lazily so it never enters nnx state.
    """

    def __init__(self, chain: Chain, dim: int, cond_dim: int):
        self.chain = chain
        self.dim = dim
        self.cond_dim = cond_dim

    def _base(self):
        return make_gaussian_prior((self.dim,))

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        base = self._base()

        def single(x_i, cond_i):
            u, logdet = self.chain.inverse(x_i, cond_i)
            return base.log_prob(u) + logdet

        if cond is None:
            return jax.vmap(lambda xi: single(xi, None))(x)
        return jax.vmap(single)(x, cond)

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None) -> Array:
        base = self._base()
        if cond is not None:
            nsamples = cond.shape[0]
        u = base.sample(key, (nsamples,))            # (nsamples, dim)

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            return jax.vmap(lambda ui: single(ui, None))(u)
        return jax.vmap(single)(u, cond)

    def set_standardization(self, mean, std) -> None:
        """Set the data-end Standardize bijection's mean/std buffers in place.

        Raises ValueError if the flow was built with ``standardize=False``.
        """
        mean = jnp.asarray(mean)
        std = jnp.asarray(std)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "Flow has no Standardize bijection (built with standardize=False).")


def make_maf(rngs, dim, cond_dim=0, n_layers=5, transformer=None,
             nn_width=64, nn_depth=2, permutation="reverse",
             standardize=True, zero_init=True) -> Flow:
    """Build an affine MAF as a stack of (MaskedAutoregressive, Permutation) layers.

    Parameters
    ----------
    rngs : nnx.Rngs
    dim : int               Target (autoregressive) dimension.
    cond_dim : int          Conditioning dim; 0 for unconditional.
    n_layers : int          Number of autoregressive layers.
    transformer : object    Elementwise transformer; defaults to ``Affine()``.
    nn_width, nn_depth : int
    permutation : str       "reverse" (alternating via stacking) or "random".
    standardize : bool      Prepend a data-end Standardize bijection (identity
                            until the pipeline sets stats).
    zero_init : bool        Identity warm-start init.
    """
    if transformer is None:
        transformer = Affine()

    bijections = []
    for i in range(n_layers):
        bijections.append(
            MaskedAutoregressive(dim, cond_dim, transformer, nn_width, nn_depth,
                                 rngs, zero_init=zero_init)
        )
        if i < n_layers - 1:
            if permutation == "reverse":
                bijections.append(Permutation.reverse(dim))
            elif permutation == "random":
                bijections.append(Permutation.random(dim, rngs))
            else:
                raise ValueError(f"unknown permutation {permutation!r}")

    if standardize:
        # data-end: appended last so it is applied first in inverse (data->noise)
        bijections.append(Standardize(dim))

    return Flow(Chain(bijections), dim=dim, cond_dim=cond_dim)
