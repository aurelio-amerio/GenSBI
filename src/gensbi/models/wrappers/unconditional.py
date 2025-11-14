from jax import Array
from typing import Optional


import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time


class UnconditionalWrapper(ModelWrapper):
    """
    Module to handle conditioning in the Unconditional estimation model.

    Args:
        model: Unconditional model instance.
    """
    def __init__(self, model):
        super().__init__(model)

    def __call__(
        self, 
        t: Array, 
        obs: Array, 
        obs_ids: Array, 
        **kwargs,
    ) -> Array:
        """
        Perform inference based on conditioning.

        Args:
            t (Array): Time steps.
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.

        Returns:
            Array: Model output.
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