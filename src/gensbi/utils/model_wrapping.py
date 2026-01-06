"""
Model wrapping utilities for GenSBI.

This module provides wrapper classes for models used in flow matching and diffusion,
facilitating integration with ODE solvers and providing utilities for computing
vector fields and divergences.
"""
from abc import ABC
from flax import nnx
from jax import Array
import jax.numpy as jnp

from typing import Callable

from .math import divergence, _expand_dims, _expand_time




class ModelWrapper(nnx.Module):
    """
    Wrapper class for models to provide ODE solver integration.
    
    This class wraps around another model and provides methods for computing
    the vector field and divergence, which are useful for ODE solvers that
    require these quantities.
    
    Parameters
    ----------
        model: The model to wrap.
    """

    def __init__(self, model: nnx.Module) -> None:
        """
        Initialize the model wrapper.
        
        Parameters
        ----------
            model: The model to wrap.
        """
        self.model = model

    def __call__(self, t: Array, obs: Array, *args, **kwargs) -> Array:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`obs` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given obs, t
        | returns the model output for input obs at time t, with extra information `extra`.

        Parameters
        ----------
            obs : Array
                input data to the model (batch_size, ...).
            t : Array
                time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns
        -------
            Array
                model output.
        """
        obs = _expand_dims(obs)
        # t = self._expand_time(t)

        return self.model(obs, t, *args, **kwargs)

    def get_vector_field(self, **kwargs) -> Callable:
        r"""Compute the vector field of the model, properly squeezed for the ODE term.

        Parameters
        ----------
            x : Array
                input data to the model (batch_size, ...).
            t : Array
                time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.

        Returns
        -------
            Array
                vector field of the model.
        """

        def vf(t, x, args):
            # merge args and kwargs
            args = args if args is not None else {}
            vf = self(t, x, **args, **kwargs)
            # squeeze the first dimension of the vector field if it is 1
            # if vf.shape[0] == 1:
            #     vf = jnp.squeeze(vf, axis=0)

            # vf = jnp.squeeze(vf, axis=-1)
            return vf

        return vf

    def get_divergence(self, **kwargs) -> Callable:
        r"""Compute the divergence of the model.

        Parameters
        ----------
            t : Array
                time (batch_size).
            x : Array
                input data to the model (batch_size, ...).
            args: additional information forwarded to the model, e.g., text condition.

        Returns
        -------
            Array
                divergence of the model.
        """
        vf = self.get_vector_field(**kwargs)

        def div_(t, x, args):
            div = divergence(vf, t, x, args)
            # squeeze the first dimension of the divergence if it is 1
            # if div.shape[0] == 1:
            #     div = jnp.squeeze(div, axis=0)
            return div

        return div_


# class GuidedModelWrapper(ModelWrapper):
#     """
#     This class is used to wrap around another model. We define a call method which returns the model output.
#     Furthermore, we define a vector_field method which computes the vector field of the model,
#     and a divergence method which computes the divergence of the model, in a form useful for diffrax.
#     This is useful for ODE solvers that require the vector field and divergence of the model.

#     """

#     cfg_scale: float

#     def __init__(self, model, cfg_scale=0.7):
#         super().__init__(model)
#         self.cfg_scale = cfg_scale

#     def __call__(self, t: Array, obs: Array, *args, **kwargs) -> Array:
#         r"""Compute the guided model output as a weighted sum of conditioned and unconditioned predictions.

#         Args:
#             obs (Array): input data to the model (batch_size, ...).
#             t (Array): time (batch_size).
#             args: additional information forwarded to the model, e.g., text condition.
#             **kwargs: additional keyword arguments.

#         Returns:
#             Array: guided model output.
#         """
#         kwargs.pop("conditioned", None)  # we set this flag manually
#         # Get outputs from parent class
#         c_out = super().__call__(t, obs, *args, conditioned=True, **kwargs)
#         u_out = super().__call__(t, obs, *args, conditioned=False, **kwargs)

#         return (1 - self.cfg_scale) * u_out + self.cfg_scale * c_out

#     def get_vector_field(self, **kwargs) -> Callable:
#         """Compute the guided vector field as a weighted sum of conditioned and unconditioned predictions."""
#         # Get vector fields from parent class
#         c_vf = super().get_vector_field(conditioned=True, **kwargs)
#         u_vf = super().get_vector_field(conditioned=False, **kwargs)

#         def g_vf(t, x, args):
#             return (1 - self.cfg_scale) * u_vf(t, x, args) + self.cfg_scale * c_vf(
#                 t, x, args
#             )

#         return g_vf
