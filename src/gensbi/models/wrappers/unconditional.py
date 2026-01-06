"""
Unconditional model wrapper for GenSBI.

This module provides a wrapper class for unconditional models used in flow matching,
handling proper input expansion and calling conventions.
"""
from jax import Array
from typing import Optional


import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time



class UnconditionalWrapper(ModelWrapper):
    """
    Wrapper for unconditional models to handle input expansion and calling convention.

    Parameters
    ----------
        model: The unconditional model instance to wrap.
    """
    def __init__(self, model):
        """
        Initialize the UnconditionalWrapper.

        Parameters
        ----------
            model: The unconditional model instance to wrap.
        """
        super().__init__(model)

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        **kwargs,
    ) -> Array:
        """
        Call the wrapped model with expanded inputs.

        Parameters
        ----------
            t : Array
                Time steps.
            obs : Array
                Observations.
            obs_ids : Array
                Observation identifiers.
            **kwargs: Additional keyword arguments passed to the model.

        Returns
        -------
            Array
                Model output.
        """

        t = _expand_time(t)
        obs = _expand_dims(obs)
        obs_ids = _expand_dims(obs_ids)
        
        return self.model(
            obs=obs,
            t=t,
            node_ids=obs_ids,
            condition_mask=jnp.zeros(obs.shape, dtype=jnp.bool_),
            **kwargs,
        )