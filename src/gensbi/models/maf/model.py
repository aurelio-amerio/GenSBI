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
from jax.typing import DTypeLike

from gensbi.core.prior import make_gaussian_prior
from gensbi.models.core.stats import fit_stat
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

    Parameters
    ----------
    rngs : nnx.Rngs
        Flax RNG container used to initialise all trainable parameters.
    dim : int
        Dimensionality of the target variable.
    cond_dim : int, optional
        Dimensionality of the conditioning input.  Default is 0 (unconditional).
    n_layers : int, optional
        Number of :class:`~gensbi.models.maf.made.MaskedAutoregressive` layers.
        Default is 5.
    transformer : Bijection or None, optional
        Elementwise bijection used by each autoregressive layer.  If ``None``
        (default), an
        :class:`~gensbi.normalizing_flows.bijections.transformers.Affine`
        bijection is constructed automatically in ``__post_init__``.
    nn_width : int, optional
        Width of each hidden layer in the MADE conditioner network.
        Default is 64.
    nn_depth : int, optional
        Number of hidden layers in the MADE conditioner network.  Default is 2.
    permutation : str, optional
        Permutation applied between autoregressive layers.  ``"reverse"``
        (default) reverses the dimension ordering; ``"random"`` applies a
        random shuffle sampled at construction time.
    standardize : bool, optional
        If ``True`` (default), append a
        :class:`~gensbi.normalizing_flows.bijections.standardize.Standardize`
        bijection at the data end of the chain.
    zero_init : bool, optional
        If ``True`` (default), zero-initialise the output layer of each MADE
        network so the flow starts as an identity transform.
    param_dtype : DTypeLike, optional
        Dtype for all stored (master) MADE kernel/bias parameters. Default is
        ``float32``.
    dtype : DTypeLike, optional
        Compute dtype knob threaded through the MADE conditioners. Default is
        ``float32`` (unlike the bf16-default DiT-family models, MAF keeps
        fp32 compute by default pending dedicated stability testing — see
        the mixed-precision design spec). Log-det accumulation is
        unconditionally fp32 regardless of this knob.
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
    channels: int = 1
    cond_channels: int = 1
    param_dtype: DTypeLike = jnp.float32
    dtype: DTypeLike = jnp.float32

    def __post_init__(self):
        if self.transformer is None:
            self.transformer = Affine()
        if self.permutation not in ("reverse", "random"):
            raise ValueError(f"unknown permutation {self.permutation!r}")


class MAFlow(nnx.Module):
    """Masked Autoregressive Flow for exact density evaluation and sampling.

    Stacks :class:`~gensbi.models.maf.made.MaskedAutoregressive` layers
    separated by permutations, with an optional data-end
    :class:`~gensbi.normalizing_flows.bijections.standardize.Standardize`
    bijection, over a standard-normal base distribution.

    Log-density follows the change-of-variables formula:
    ``log_prob(x, cond) = base.log_prob(u) + logdet``, where
    ``u, logdet = chain.inverse(x, cond)``.  The base distribution is built
    lazily and never enters ``nnx`` state.

    Parameters
    ----------
    params : MAFlowParams
        Full architecture configuration; see :class:`MAFlowParams`.
    """

    def __init__(self, params: MAFlowParams):
        rngs = params.rngs
        dim = params.dim
        self.channels = params.channels
        self.cond_channels = params.cond_channels
        flat_dim = dim * self.channels
        flat_cond_dim = params.cond_dim * self.cond_channels
        bijections = []
        for i in range(params.n_layers):
            bijections.append(
                MaskedAutoregressive(flat_dim, flat_cond_dim, params.transformer,
                                     params.nn_width, params.nn_depth, rngs,
                                     zero_init=params.zero_init,
                                     param_dtype=params.param_dtype,
                                     dtype=params.dtype))
            if i < params.n_layers - 1:
                if params.permutation == "reverse":
                    bijections.append(Permutation.reverse(flat_dim))
                else:
                    bijections.append(Permutation.random(flat_dim, rngs))
        if params.standardize:
            bijections.append(Standardize(flat_dim))
        self.chain = Chain(bijections)
        self.dim = dim
        self.flat_dim = flat_dim
        self.cond_dim = params.cond_dim

    def _base(self):
        return make_gaussian_prior((self.flat_dim,))

    def log_prob(self, x: Array, cond: Array | None = None) -> Array:
        """Compute the change-of-variables log-density for a batch of samples.

        Parameters
        ----------
        x : Array
            Data batch.  Shape ``(B, dim)`` when ``channels == 1``, or
            ``(B, dim, C)`` when ``channels > 1`` (the channel axis is
            flattened internally to ``(B, dim * C)``).
        cond : Array or None, optional
            Conditioning batch of shape ``(B, cond_dim)`` for
            ``cond_channels == 1``, or ``(B, cond_dim, C_cond)`` for
            ``cond_channels > 1`` (also flattened internally).  Pass
            ``None`` for an unconditional model.

        Returns
        -------
        Array
            Log-probability of shape ``(B,)``.
        """
        base = self._base()
        x = jnp.asarray(x)
        x = x.reshape(x.shape[0], -1)
        if cond is not None:
            cond = jnp.asarray(cond)
            cond = cond.reshape(cond.shape[0], -1)

        def single(x_i, cond_i):
            u, logdet = self.chain.inverse(x_i, cond_i)
            return base.log_prob(u) + logdet

        if cond is None:
            return jax.vmap(lambda xi: single(xi, None))(x)
        return jax.vmap(single)(x, cond)

    def sample(self, key, cond: Array | None = None, nsamples: int | None = None) -> Array:
        """Draw samples from the flow.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        cond : Array or None, optional
            Conditioning batch of shape ``(nsamples, cond_dim)`` for
            ``cond_channels == 1``, or ``(nsamples, cond_dim, C_cond)`` for
            ``cond_channels > 1`` (flattened internally).  If provided, the
            number of samples is inferred from ``cond.shape[0]`` and
            ``nsamples`` is ignored.
        nsamples : int or None, optional
            Number of samples to draw.  Required when ``cond`` is ``None``.

        Returns
        -------
        Array
            Sample array of shape ``(nsamples, dim, channels)`` for all ``C >= 1``
            (``C = 1`` gives ``(nsamples, dim, 1)``; channel axis is never collapsed).
        """
        base = self._base()
        if cond is not None:
            cond = jnp.asarray(cond)
            cond = cond.reshape(cond.shape[0], -1)
            nsamples = cond.shape[0]
        u = base.sample(key, (nsamples,))

        def single(u_i, cond_i):
            x, _ = self.chain.forward(u_i, cond_i)
            return x

        if cond is None:
            x = jax.vmap(lambda ui: single(ui, None))(u)
        else:
            x = jax.vmap(single)(u, cond)
        x = x.reshape(x.shape[0], self.dim, self.channels)   # always carry the channel
        return x

    def set_standardization(self, mean, std) -> None:
        """Set the data-end Standardize bijection's mean/std buffers in place.

        Accepts shapes ``(dim,)`` (broadcast to ``(dim, 1)``), ``(dim, 1)``,
        ``(C,)`` (per-channel broadcast), or a scalar broadcastable to
        ``(dim, channels)``.

        Raises ValueError if built with ``standardize=False``.
        """
        es = (self.dim, self.channels)
        mean = fit_stat(mean, es).reshape(-1)
        std = fit_stat(std, es).reshape(-1)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
