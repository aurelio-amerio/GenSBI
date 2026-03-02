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

from .math import divergence, divergence_hutchinson, _expand_dims, _expand_time

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
            # Filter out divergence-only keys (e.g. div_v for Hutchinson)
            # that are not model parameters.
            _DIVERGENCE_KEYS = {"div_v"}
            model_args = {k: v for k, v in args.items() if k not in _DIVERGENCE_KEYS}
            vf = self(t, x, **model_args, **kwargs)
            return vf

        return vf

    def get_divergence(self, exact: bool = True, **kwargs) -> Callable:
        r"""Return a function that computes the divergence of the vector field.

        Parameters
        ----------
            exact : bool
                If ``True`` (default), compute the exact divergence via
                the full Jacobian (``jax.jacfwd`` + trace).  If ``False``,
                use the Hutchinson stochastic trace estimator (single JVP
                with a Rademacher probe).  The Hutchinson variant requires
                the probe vector to be passed at call time inside
                ``args["div_v"]``.
            **kwargs
                Static keyword arguments forwarded to ``get_vector_field``.

        Returns
        -------
            Callable
                ``div_(t, x, args)`` — divergence function compatible with
                diffrax ODE terms.
        """
        vf = self.get_vector_field(**kwargs)

        if exact:
            def div_(t, x, args):
                return divergence(vf, t, x, args)
        else:
            def div_(t, x, args):
                args = dict(args)  # shallow copy to avoid mutating the caller's dict
                v = args.pop("div_v")
                return divergence_hutchinson(vf, t, x, args, v=v)

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
