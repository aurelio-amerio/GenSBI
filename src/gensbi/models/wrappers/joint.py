from jax import Array
from typing import Optional


import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time


class JointWrapper(ModelWrapper):
    """
    Module to handle conditioning in the Joint estimation model.

    Args:
        model: Joint model model instance.
    """
    def __init__(self, model):
        super().__init__(model)

    def conditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        t: Array, 
        **kwargs,
    ) -> Array:
        """
        Perform conditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            t (Array): Time steps.

        Returns:
            Array: Conditioned output.
        """
        
        obs_dim = obs.shape[1]
        cond_dim = cond.shape[1]
        # repeat cond on the first dimension to match obs
        cond = jnp.broadcast_to(
            cond, (obs.shape[0], *cond.shape[1:])
        )

        condition_mask_dim = obs_dim + cond_dim

        condition_mask = jnp.zeros((condition_mask_dim,), dtype=jnp.bool_)
        condition_mask = condition_mask.at[obs_dim:].set(True)

        x = jnp.concatenate([obs, cond], axis=1)
        node_ids = jnp.concatenate([obs_ids, cond_ids], axis=1)

        res = self.model(
            obs=x,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            **kwargs,
        )
        # now return only the values on which we are not conditioning
        res = res[:, :obs_dim]
        return res

    def unconditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        t: Array, 
        **kwargs,
    ) -> Array:
        """
        Perform unconditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            t (Array): Time steps.

        Returns:
            Array: Unconditioned output.
        """

        condition_mask = jnp.zeros((obs.shape[1],), dtype=jnp.bool_)

        node_ids = obs_ids

        res = self.model(
            obs=obs,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            **kwargs,
        )

        return res

    def __call__(
        self, 
        t: Array, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        conditioned: bool = True, 
        **kwargs,
    ) -> Array:
        """
        Perform inference based on conditioning.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            timesteps (Array): Time steps.
            conditioned (bool): Whether to perform conditioned inference.

        Returns:
            Array: Model output.
        """
        t = _expand_time(t)
        obs = _expand_dims(obs)
        cond = _expand_dims(cond)
        
        obs_ids = _expand_dims(obs_ids)
        cond_ids = _expand_dims(cond_ids)
        
        if conditioned:
            return self.conditioned(obs, obs_ids, cond, cond_ids, t, **kwargs)
        else:
            return self.unconditioned(obs, obs_ids, t, **kwargs)