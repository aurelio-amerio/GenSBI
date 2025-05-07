# FIXME: first pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from flax import nnx
from jax import Array
import jax.numpy as jnp

from .utils import divergence


class ModelWrapper(nnx.Module):
    """
    This class is used to wrap around another model. We define a call method which returns the model output. 
    Furthermore, we define a vector_field method which computes the vector field of the model,
    and a divergence method which computes the divergence of the model, in a form useful for diffrax.
    This is useful for ODE solvers that require the vector field and divergence of the model.

    """

    def __init__(self, model: nnx.Module):
        self.model = model

    def __call__(self, x: Array, t: Array, **extras) -> Array:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: model output.
        """
        return self.model(x=x, t=t, **extras)

    def vector_field(self, t: Array, x: Array, args) -> Array:
        r"""Compute the vector field of the model, properly squeezed for the ODE term.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: vector field of the model.
        """
        return jnp.squeeze(self(x, t, **args))

    def divergence(self, t: Array, x: Array, args=None) -> Array:
        r"""Compute the divergence of the model.

        Args:
            t (Array): time (batch_size).
            x (Array): input data to the model (batch_size, ...).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: divergence of the model.
        """
        vf_wrapped = lambda x, t: self.vector_field(t, x, args)

        return jnp.squeeze(divergence(vf_wrapped, x, t))
