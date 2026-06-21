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
from gensbi.normalizing_flows.bijections.masked_linear import MaskedLinear
from gensbi.normalizing_flows.bijections.masks import make_mask


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
    dim : int               Autoregressive (target) dimension.
    cond_dim : int          Conditioning dimension; 0 for unconditional.
    num_params : int        Transform params per dim (Affine: 2).
    nn_width, nn_depth : int
    rngs : nnx.Rngs
    zero_init : bool        Identity warm-start: zero the output layer so all
                            transform params start at 0 (Affine -> identity).
                            Default True; tests pass False so the net is live.
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
            self.output_layer.linear.kernel.value = jnp.zeros_like(
                self.output_layer.linear.kernel.value)
            self.output_layer.linear.bias.value = jnp.zeros_like(
                self.output_layer.linear.bias.value)

    def __call__(self, x: Array, cond: Array | None = None) -> Array:
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
    """MADE conditioner + elementwise transformer = one autoregressive flow step.

    inverse (data->noise) is one MADE pass (fast); forward (noise->data) is a
    sequential ``lax.scan`` over dims (slow).
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
        params = self.made(x, cond)              # (dim, num_params), single pass
        return self.transformer.inverse(x, params)

    def forward(self, u: Array, cond: Array | None = None):
        def body(x, i):
            params = self.made(x, cond)
            x_i = self.transformer.forward_dim(u[i], params[i])
            return x.at[i].set(x_i), None

        x0 = jnp.zeros_like(u)
        x, _ = jax.lax.scan(body, x0, jnp.arange(self.dim))
        # log-det from the completed x (forward logdet = +sum(a))
        params = self.made(x, cond)
        _, logdet = self.transformer.forward(u, params)
        return x, logdet
