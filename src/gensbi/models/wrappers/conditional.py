"""
Conditional model wrapper for GenSBI.

This module provides a wrapper class for conditional models used in flow matching,
handling proper input expansion and calling conventions for conditional inference.
"""

from jax import Array


from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time




class ConditionalWrapper(ModelWrapper):
    """
    Wrapper for conditional models to handle input expansion and calling convention.

    Parameters
    ----------
        model: The conditional model instance to wrap.
    """
    def __init__(self, model):
        """
        Initialize the ConditionalWrapper.

        Parameters
        ----------
            model: The conditional model instance to wrap.
        """
        super().__init__(model)

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
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
            cond : Array
                Conditioning values.
            cond_ids : Array
                Conditioning identifiers.
            conditioned : bool | Array, optional
                Whether to use conditioning. Defaults to True.
            guidance : Array | None, optional
                Optional guidance input.

        Returns
        -------
            Array
                Model output.
        """
        obs = _expand_dims(obs)
        t = _expand_time(t)
        cond = _expand_dims(cond)
        obs_ids = _expand_dims(obs_ids)
        cond_ids = _expand_dims(cond_ids)

        return self.model(
            obs=obs,
            t=t,
            cond=cond,
            obs_ids=obs_ids,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
            **kwargs,
        )
