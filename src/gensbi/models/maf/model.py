"""MAF: affine/spline masked-autoregressive normalizing flow.

Self-contained density model (absorbs the former ``Flow`` container and the
``make_maf`` factory). Builds a Chain of (MaskedAutoregressive, Permutation)
layers + an optional data-end Standardize, over a standard-normal base.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows.bijections.base import Bijection
from gensbi.models.maf.made import MaskedAutoregressive
from gensbi.normalizing_flows.bijections.chain import Chain
from gensbi.normalizing_flows.bijections.permutation import Permutation
from gensbi.normalizing_flows.bijections.standardize import Standardize
from gensbi.normalizing_flows.bijections.transformers import Affine


@dataclass
class MAFlowParams:
    """Architecture parameters for :class:`MAFlow`.

    Only ``rngs`` and ``dim`` are required. ``transformer`` defaults to
    ``Affine()`` (pass ``RQSpline()`` for a spline flow).
    """

    rngs: nnx.Rngs
    dim: int
    cond_dim: int = 0
    n_layers: int = 5
    transformer: Bijection | None = None
    nn_width: int = 64
    nn_depth: int = 2
    permutation: str = "reverse"
    standardize: bool = True
    zero_init: bool = True

    def __post_init__(self):
        if self.transformer is None:
            self.transformer = Affine()
        if self.permutation not in ("reverse", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")


class MAFlow(nnx.Module):
    """Affine/spline MAF over ``(batch, dim)`` data, optionally conditioned.

    ``log_prob(x, cond) = base.log_prob(u) + logdet`` with
    ``u, logdet = chain.inverse(x, cond)``; the base is a standard normal over
    ``(dim,)`` built lazily so it never enters nnx state.
    """

    def __init__(self, params: MAFlowParams):
        rngs = params.rngs
        dim = params.dim
        bijections = []
        for i in range(params.n_layers):
            bijections.append(
                MaskedAutoregressive(dim, params.cond_dim, params.transformer,
                                     params.nn_width, params.nn_depth, rngs,
                                     zero_init=params.zero_init))
            if i < params.n_layers - 1:
                if params.permutation == "reverse":
                    bijections.append(Permutation.reverse(dim))
                else:
                    bijections.append(Permutation.random(dim, rngs))
        if params.standardize:
            bijections.append(Standardize(dim))
        self.chain = Chain(bijections)
        self.dim = dim
        self.cond_dim = params.cond_dim

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
        u = base.sample(key, (nsamples,))

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            return jax.vmap(lambda ui: single(ui, None))(u)
        return jax.vmap(single)(u, cond)

    def set_standardization(self, mean, std) -> None:
        """Set the data-end Standardize bijection's mean/std buffers in place.

        Raises ValueError if built with ``standardize=False``.
        """
        mean = jnp.asarray(mean)
        std = jnp.asarray(std)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
