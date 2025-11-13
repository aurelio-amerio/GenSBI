

from jax import Array


from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time



class ConditionalWrapper(ModelWrapper):
    def __init__(self, model):
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
    ) -> Array:

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
        )
