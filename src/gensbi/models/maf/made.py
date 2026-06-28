"""MADE conditioner with concatenation-based conditioning (flowjax-style).

Conditioning variables are concatenated onto the input and given autoregressive
rank -1 (below every data dimension), so every output -- including the first --
may depend on the condition while the condition depends on nothing. This is the
standard conditional-MAF approach (Papamakarios et al. 2017; flowjax). NO
cross-feature normalisation (LayerNorm/RMSNorm/GroupNorm): MADE hidden units
carry the autoregressive rank, so cross-unit statistics would mix ranks and
silently break the flow. See spec §6.

The conditioner is a single cohesive module behind the
``(x, cond) -> (dim, num_params)`` interface; alternative conditioning schemes
(FiLM, T-NAF, ...) may be added later as drop-in conditioners.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from gensbi.normalizing_flows.bijections.base import Bijection
from gensbi.models.maf.masked_linear import MaskedLinear
from gensbi.models.maf.masks import make_mask


def _rank_vectors(dim, nn_width, num_params, cond_dim):
    """0-indexed MADE ranks for input, hidden, and output units.

    With conditioning (``cond_dim > 0``) the conditioning inputs get rank -1
    (before every data dim) and hidden ranks are shifted into ``[-1, dim-2]`` so
    some hidden units carry only the condition and can feed output dim 0.
    """
    out_ranks = jnp.repeat(jnp.arange(dim), num_params)
    if cond_dim > 0:
        in_ranks = jnp.concatenate(
            [jnp.arange(dim), -jnp.ones(cond_dim, dtype=jnp.int32)])
        hidden_ranks = (jnp.arange(nn_width) % dim) - 1
    elif dim > 1:
        in_ranks = jnp.arange(dim)
        hidden_ranks = jnp.arange(nn_width) % (dim - 1)
    else:
        in_ranks = jnp.arange(dim)
        hidden_ranks = jnp.zeros(nn_width, dtype=jnp.int32)
    return in_ranks, hidden_ranks, out_ranks


class MADE(nnx.Module):
    """Autoregressive conditioner: ``(x, cond) -> params`` of shape ``(dim, num_params)``.

    Conditioning is by concatenation: ``cond`` is appended to ``x`` and given
    autoregressive rank -1, so every output (incl. dim 0) may depend on it.

    Parameters
    ----------
    dim : int
        Autoregressive (target) dimension.
    cond_dim : int
        Conditioning dimension; 0 for unconditional.
    num_params : int
        Transform parameters per dimension (e.g. 2 for an Affine transformer).
    nn_width : int
        Width of each masked hidden layer.
    nn_depth : int
        Number of masked hidden layers.
    rngs : nnx.Rngs
        Flax RNG container for parameter initialisation.
    zero_init : bool, optional
        If ``True`` (default), zero-initialise the output layer so that all
        transform parameters start at 0 (Affine becomes the identity).
    param_dtype : DTypeLike, optional
        Dtype for all kernel and bias parameters.  Default is ``float32``.
    activation : Callable, optional
        Element-wise activation applied after each hidden layer.
        Default is :func:`jax.nn.silu`.
    """

    def __init__(self, dim, cond_dim, num_params, nn_width, nn_depth, rngs,
                 zero_init: bool = True, param_dtype: DTypeLike = jnp.float32,
                 activation=jax.nn.silu):
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_params = num_params
        self.activation = activation

        in_ranks, hidden_ranks, out_ranks = _rank_vectors(
            dim, nn_width, num_params, cond_dim)
        in_mask = make_mask(in_ranks, hidden_ranks, strict=False)
        hidden_mask = make_mask(hidden_ranks, hidden_ranks, strict=False)
        out_mask = make_mask(hidden_ranks, out_ranks, strict=True)

        self.input_layer = MaskedLinear(dim + cond_dim, nn_width, in_mask,
                                        rngs=rngs, param_dtype=param_dtype)
        self.hidden_layers = nnx.List([
            MaskedLinear(nn_width, nn_width, hidden_mask, rngs=rngs,
                         param_dtype=param_dtype)
            for _ in range(nn_depth)
        ])
        self.output_layer = MaskedLinear(nn_width, dim * num_params, out_mask,
                                         rngs=rngs, param_dtype=param_dtype)
        if zero_init:
            # Identity warm-start: zero output params -> affine is identity.
            self.output_layer.linear.kernel[...] = jnp.zeros_like(
                self.output_layer.linear.kernel[...])
            self.output_layer.linear.bias[...] = jnp.zeros_like(
                self.output_layer.linear.bias[...])

    def __call__(self, x: Array, cond: Array | None = None) -> Array:
        """Compute the transform-parameter array from input and optional conditioning.

        Parameters
        ----------
        x : Array
            Data input of shape ``(dim,)``.
        cond : Array or None, optional
            Conditioning input of shape ``(cond_dim,)``, or ``None`` for an
            unconditional conditioner.  Required when ``cond_dim > 0``.

        Returns
        -------
        Array
            Transform-parameter array of shape ``(dim, num_params)``.

        Raises
        ------
        ValueError
            If ``cond_dim > 0`` and ``cond`` is ``None``.
        """
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    "cond is required: this MADE was built with cond_dim > 0")
            nn_input = jnp.concatenate([x, cond])
        else:
            nn_input = x
        h = self.activation(self.input_layer(nn_input))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        out = self.output_layer(h)
        return out.reshape(self.dim, self.num_params)


class MaskedAutoregressive(Bijection):
    """MADE conditioner coupled with an elementwise transformer: one MAF layer.

    Implements the :class:`~gensbi.normalizing_flows.bijections.base.Bijection`
    contract.  :meth:`inverse` maps data to noise in a single parallel MADE
    pass (fast); :meth:`forward` maps noise to data via a sequential
    ``lax.scan`` over dimensions (slow).

    Parameters
    ----------
    dim : int
        Dimensionality of the target variable.
    cond_dim : int
        Dimensionality of the conditioning input; 0 for unconditional.
    transformer : Bijection
        Elementwise bijection (e.g.
        :class:`~gensbi.normalizing_flows.bijections.transformers.Affine`)
        whose parameters are predicted by the MADE network.
    nn_width : int
        Width of each hidden layer in the MADE network.
    nn_depth : int
        Number of hidden layers in the MADE network.
    rngs : nnx.Rngs
        Flax RNG container for parameter initialisation.
    zero_init : bool, optional
        If ``True`` (default), zero-initialise the MADE output layer so that
        the flow starts as an identity transform.
    """

    def __init__(self, dim, cond_dim, transformer, nn_width, nn_depth, rngs,
                 zero_init: bool = True):
        self.dim = dim
        self.transformer = transformer
        self.made = MADE(dim=dim, cond_dim=cond_dim,
                         num_params=transformer.num_params,
                         nn_width=nn_width, nn_depth=nn_depth,
                         zero_init=zero_init, rngs=rngs)

    def inverse(self, x: Array, cond: Array | None = None):
        """Map data to noise (the density-evaluation direction).

        Runs a single parallel MADE forward pass to obtain the transform
        parameters, then applies the elementwise transformer inverse to ``x``.

        Parameters
        ----------
        x : Array
            Data-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional map.

        Returns
        -------
        u : Array
            Noise-space output of shape ``(dim,)``.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the inverse map.
        """
        params = self.made(x, cond)              # (dim, num_params), single pass
        return self.transformer.inverse(x, params)

    def forward(self, u: Array, cond: Array | None = None):
        """Map noise to data (the sampling direction).

        Runs a sequential ``lax.scan`` over dimensions: each step calls the
        MADE network on the partially-built output to obtain parameters for
        the next dimension.  Because dimension ``i``'s parameters depend only on
        dimensions ``< i`` (already final), the per-dimension log-determinant is
        accumulated inside the scan, avoiding a second full MADE pass.

        Parameters
        ----------
        u : Array
            Noise-space input of shape ``(dim,)``.
        cond : Array or None, optional
            Conditioning input, or ``None`` for an unconditional map.

        Returns
        -------
        x : Array
            Data-space output of shape ``(dim,)``.
        logabsdet : Array
            Log absolute determinant of the Jacobian of the forward map.
        """
        def body(x, i):
            params = self.made(x, cond)
            # params[i] depends only on x[<i] (already final), so x_i and its
            # logdet contribution are final at step i.
            x_i, logdet_i = self.transformer.forward_dim(u[i], params[i])
            return x.at[i].set(x_i), logdet_i

        x0 = jnp.zeros_like(u)
        x, logdet_steps = jax.lax.scan(body, x0, jnp.arange(self.dim))
        logdet = jnp.sum(logdet_steps)                    # forward logdet = +sum(a)
        return x, logdet
