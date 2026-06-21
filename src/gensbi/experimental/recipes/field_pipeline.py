"""Field-shaped conditional pipeline (experimental).

``ConditionalPipeline`` assumes token-shaped observations ``(B, dim, ch)``:
it flattens ``dim_obs``, resolves embedding ids, and builds a rank-2 prior.
Pixel-space field models (FieldDiT) need ``(B, H, W, C)`` observations, a
rank-3 ``event_shape``, no external ids, and rank-aware input expansion.
"""

import jax.numpy as jnp

from gensbi.core.generative_method import GenerativeMethod
from gensbi.recipes.conditional_pipeline import ConditionalPipeline
from gensbi.recipes.pipeline import AbstractPipeline
from gensbi.utils.model_wrapping import ModelWrapper, _expand_time


class FieldConditionalWrapper(ModelWrapper):
    """Wrapper for field-shaped conditional models.

    Expansion is event-rank-aware (a field event is rank 3, ``(H, W, C)``):

    - ``obs`` with ``ndim == 3`` is treated as unbatched -> ``(1, H, W, C)``.
    - ``cond`` with ``ndim == 1`` (``(k,)``) -> ``(1, k)``; 2D+ cond is
      assumed batched (``(B, k)`` / ``(B, k, c)``).
    - batch-1 ``cond`` is broadcast to the obs batch (sampling N draws for a
      single ``x_o``; the model itself deliberately does not broadcast).
    - ids are passed through untouched (FieldDiT builds rope ids internally).
    """

    def __init__(self, model):
        super().__init__(model)

    def __call__(
        self,
        t,
        obs,
        cond,
        obs_ids=None,
        cond_ids=None,
        conditioned=True,
        guidance=None,
        **kwargs,
    ):
        if obs.ndim == 3:
            obs = obs[None, ...]
        if cond.ndim == 1:
            cond = cond[None, ...]
        if cond.shape[0] == 1 and obs.shape[0] > 1:
            cond = jnp.repeat(cond, obs.shape[0], axis=0)
        t = _expand_time(t)

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


class FieldConditionalPipeline(ConditionalPipeline):
    """Conditional pipeline for pixel-space fields ``(B, H, W, C)``.

    Differences from :class:`ConditionalPipeline`:

    - ``event_shape = (*field_shape, ch_obs)`` — prior, path, and sampling are
      field-shaped (``sample`` returns ``(nsamples, H, W, C)``).
    - no obs/cond id resolution: ``obs_ids``/``cond_ids`` are ``None`` in all
      extras; the model builds its rope ids internally and ignores them.
    - :class:`FieldConditionalWrapper` for event-rank-aware expansion.

    Datasets must yield ``(obs, cond)`` batches with ``obs`` of shape
    ``(B, H, W, C)`` and ``cond`` of shape ``(B, k)`` or ``(B, k, c)``.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        field_shape,
        dim_cond,
        method: GenerativeMethod,
        ch_obs=1,
        ch_cond=1,
        params=None,
        training_config=None,
    ):
        self.method = method

        if training_config is None:
            training_config = self.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)

        # bypass ConditionalPipeline.__init__ (id resolution + rank-2 path):
        # AbstractPipeline handles datasets/EMA/optimizer/training config
        AbstractPipeline.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=tuple(field_shape),
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        self.field_shape = tuple(field_shape)
        self.event_shape = (*self.field_shape, ch_obs)
        self.obs_ids = None
        self.cond_ids = None

        self.path = method.build_path(self.training_config, event_shape=self.event_shape)
        self.loss_obj = method.build_loss(self.path)

    def _wrap_model(self):
        self.model_wrapped = FieldConditionalWrapper(self.model)
        self.ema_model_wrapped = FieldConditionalWrapper(self.ema_model)
