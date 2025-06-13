# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import jax
from jax import Array
import jax.numpy as jnp

from flow_matching.path import MixtureDiscreteProbPath


class MixturePathGeneralizedKL:
    r"""A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model :math:`p_{1|t}` w.r.t. a probability path given by ``path``. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space :math:`\mathcal{S} = \mathcal{T}^d`, :math:`\mathcal{T} = [K] = \set{1,2,\ldots,K}`, the loss is given by

    .. math::
            \ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \biggr[  p_{1|t}(x_t^i|x_t) -\delta_{x^i_1}(x_t^i) + (1-\delta_{x^i_1}(x_t^i))\left(\log p_{1|t}(x_1^i|x_t)\right)\biggr],

    where :math:`\kappa_t` is the scheduler associated with ``path``.

    Args:
        path (MixtureDiscreteProbPath): Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path: MixtureDiscreteProbPath, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        self.path = path
        self.reduction = reduction

    def __call__(self, logits: Array, x_1: Array, x_t: Array, t: Array) -> Array:
        r"""Evaluates the generalized KL loss.

        Args:
            logits (Array): posterior model output (i.e., softmax(``logits``) :math:`=p_{1|t}(x|x_t)`), shape (batch, d, K).
            x_1 (Array): target data point :math:`x_1 \sim q`, shape (batch, d).
            x_t (Array): conditional sample at :math:`x_t \sim p_t(\cdot|x_1)`, shape (batch, d).
            t (Array): times in :math:`[0,1]`, shape (batch).

        Raises:
            ValueError: reduction value must be one of ``'none'`` | ``'mean'`` | ``'sum'``.

        Returns:
            Array: Generalized KL loss.
        """
        x_1_shape = x_1.shape

        # extract x_1 value of log(p_{1|t}(x|x_t)).
        log_p_1t = jax.nn.log_softmax(logits, axis=-1)
        # JAX equivalent of torch.gather - use advanced indexing
        batch_indices = jnp.arange(x_1.shape[0])[:, None]
        d_indices = jnp.arange(x_1.shape[1])[None, :]
        log_p_1t_x1 = log_p_1t[batch_indices, d_indices, x_1]

        # extract x_t value of p_{1|t}(x|x_t).
        p_1t = jnp.exp(log_p_1t)
        p_1t_xt = p_1t[batch_indices, d_indices, x_t]

        scheduler_output = self.path.scheduler(t)

        # Create broadcasting shape for jump_coefficient
        jump_coefficient = scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
        # Reshape to broadcast properly: (batch,) -> (batch, 1, 1, ...)
        expand_dims = (slice(None),) + (None,) * (x_1.ndim - 1)
        jump_coefficient = jump_coefficient[expand_dims]
        # Tile to match x_1 shape
        tile_shape = (1,) + x_1_shape[1:]
        jump_coefficient = jnp.tile(jump_coefficient, tile_shape)
        
        delta_x1_xt = (x_t == x_1).astype(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return jnp.mean(loss)
        elif self.reduction == "sum":
            return jnp.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")