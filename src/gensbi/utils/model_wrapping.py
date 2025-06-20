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

from .math import divergence


class ModelWrapper(nnx.Module):
    """
    This class is used to wrap around another model. We define a call method which returns the model output. 
    Furthermore, we define a vector_field method which computes the vector field of the model,
    and a divergence method which computes the divergence of the model, in a form useful for diffrax.
    This is useful for ODE solvers that require the vector field and divergence of the model.

    """

    def __init__(self, model: nnx.Module):
        self.model = model

    def _call_model(self, x: Array, t: Array, args, **kwargs) -> Array:
        r"""
        This method is a wrapper around the model's call method. It allows us to pass additional arguments
        to the model, such as text conditions or other auxiliary information.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.
            **kwargs: additional keyword arguments.

        Returns:
            Array: model output.
        """
        return self.model(x, t, args=args, **kwargs)

    def __call__(self, x: Array, t: Array, args=None, **kwargs) -> Array:
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
        return self._call_model(x, t, args, **kwargs)

    def get_vector_field(self, **kwargs) -> Array:
        r"""Compute the vector field of the model, properly squeezed for the ODE term.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: vector field of the model.
        """
        def vf(t, x, args):
            vf = self._call_model(x, t, args, **kwargs)
            # squeeze the first dimension of the vector field if it is 1
            if vf.shape[0] == 1:
                vf = jnp.squeeze(vf, axis=0)
            return vf
        return vf
    

    def get_divergence(self, **kwargs) -> Array:
        r"""Compute the divergence of the model.

        Args:
            t (Array): time (batch_size).
            x (Array): input data to the model (batch_size, ...).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: divergence of the model.
        """
        vf = self.get_vector_field(**kwargs)
        def div_(t, x, args):
            div = divergence(vf, t, x, args)
            # squeeze the first dimension of the divergence if it is 1
            if div.shape[0] == 1:
                div = jnp.squeeze(div, axis=0)
            return div

        
        return div_
        

class GuidedModelWrapper(ModelWrapper):
    """
    This class is used to wrap around another model. We define a call method which returns the model output. 
    Furthermore, we define a vector_field method which computes the vector field of the model,
    and a divergence method which computes the divergence of the model, in a form useful for diffrax.
    This is useful for ODE solvers that require the vector field and divergence of the model.

    """
    cfg_scale: float 

    def __init__(self, model, cfg_scale=0.7):
        super().__init__(model)
        self.cfg_scale = cfg_scale

    def __call__(self, x: Array, t: Array, args=None, **kwargs) -> Array:
        r"""Compute the guided model output as a weighted sum of conditioned and unconditioned predictions.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.
            **kwargs: additional keyword arguments.

        Returns:
            Array: guided model output.
        """
        # Get outputs from parent class
        c_out = self._call_model(x, t, args, conditioned=True, **kwargs)
        u_out = self._call_model(x, t, args, conditioned=False, **kwargs)

        return (1 - self.cfg_scale) * u_out + self.cfg_scale * c_out

    def get_vector_field(self, **kwargs) -> Array:
        """Compute the guided vector field as a weighted sum of conditioned and unconditioned predictions."""
        # Get vector fields from parent class
        c_vf = super().get_vector_field(conditioned=True, **kwargs)
        u_vf = super().get_vector_field(conditioned=False, **kwargs)

        def g_vf(t, x, args):
            return (1 - self.cfg_scale) * u_vf(t, x, args) + self.cfg_scale * c_vf(t, x, args)
        
        return g_vf
    
        
