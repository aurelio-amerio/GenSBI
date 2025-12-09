from jax import Array
from typing import Optional


import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time


class JointWrapper(ModelWrapper):
    """
    Wrapper for joint models to handle both conditioned and unconditioned inference.

    Args:
        model: The joint model instance to wrap.
        conditioned (bool, optional): Whether to use conditioning by default. Defaults to True.
    """
    def __init__(self, model):
        """
        Initialize the JointWrapper.

        Args:
            model: The joint model instance to wrap.
            conditioned (bool, optional): Whether to use conditioning by default. Defaults to True.
        """
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
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            Array: Conditioned output (only for unconditioned variables).
        """
        obs_dim = obs.shape[1]
        cond_dim = cond.shape[1]
        cond = jnp.broadcast_to(cond, (obs.shape[0], *cond.shape[1:]))
        condition_mask_dim = obs_dim + cond_dim
        condition_mask = jnp.zeros((condition_mask_dim,), dtype=jnp.bool_)
        condition_mask = condition_mask.at[obs_dim:].set(True)
        condition_mask = condition_mask.reshape(1, condition_mask_dim, 1)
        x = jnp.concatenate([obs, cond], axis=1)
        node_ids = jnp.concatenate([obs_ids, cond_ids], axis=1)
        res = self.model(
            obs=x,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            **kwargs,
        )
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
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            Array: Unconditioned output.
        """
        condition_mask = jnp.zeros(obs.shape, dtype=jnp.bool_)
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
        Call the wrapped model for either conditioned or unconditioned inference.

        Args:
            t (Array): Time steps.
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            conditioned (bool, optional): Whether to use conditioning. If None, uses the default set at initialization.
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            Array: Model output.
        """
        t = _expand_time(t)
        obs = _expand_dims(obs)
        cond = _expand_dims(cond)

        if conditioned:
            return self.conditioned(obs, obs_ids, cond, cond_ids, t, **kwargs)
        else:
            return self.unconditioned(obs, obs_ids, t, **kwargs)