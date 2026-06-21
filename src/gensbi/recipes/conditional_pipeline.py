"""
Pipeline for training and using a Conditional model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau

from numpyro import distributions as dist
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp

from typing import Union, Tuple

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler


from gensbi.models import ConditionalWrapper

from einops import repeat

from gensbi.models.flux1 import model
from gensbi.utils.model_wrapping import _expand_dims

import os

import yaml

from gensbi.recipes.pipeline import AbstractPipeline

from gensbi.recipes.utils import (
    _resolve_embedding_ids,
    _normalize_patch_size,
    build_edm_path,
    build_sm_path,
)

import warnings


# ---------------------------------------------------------------------------
# Unified ConditionalPipeline (Phase 2)
# ---------------------------------------------------------------------------

from gensbi.core.generative_method import GenerativeMethod


class ConditionalPipeline(AbstractPipeline):
    """Model-agnostic conditional pipeline parameterized by a ``GenerativeMethod``.

    Unlike the old method-specific pipeline classes, this class
    works with **any** generative method and **any** user-provided model that
    conforms to the ``ConditionalWrapper`` interface.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : iterable
        Training dataset yielding ``(obs, cond)`` batches.
    val_dataset : iterable
        Validation dataset yielding ``(obs, cond)`` batches.
    dim_obs : int or tuple of int
        Dimension of the observation/parameter space.
    dim_cond : int or tuple of int
        Dimension of the conditioning space.
    method : GenerativeMethod
        Strategy object (e.g. ``FlowMatchingMethod()``,
        ``DiffusionEDMMethod()``, ``ScoreMatchingMethod()``).
    ch_obs : int, optional
        Number of channels per observation token. Default is 1.
    ch_cond : int, optional
        Number of channels per conditioning token. Default is 1.
    id_embedding_strategy : tuple of str, optional
        Embedding strategy for observation and conditioning IDs.
        Default is ``("absolute", "absolute")``.
    size : int or tuple of int, optional
        Patch edge length for 2D ID-embedding strategies. Default is ``2``.
        A single ``int`` is broadcast to both obs and cond (``8 -> (8, 8)``).
        A length-2 tuple ``(obs_size, cond_size)`` lets the two inputs differ.
        Use ``1`` to disable patchification for an input. Ignored for 1D
        strategies (``"absolute"``, ``"pos1d"``, ``"rope1d"``).
    params : optional
        Model parameters (stored but not used directly).
    training_config : dict, optional
        Training configuration. If ``None``, uses defaults augmented
        by ``method.get_extra_training_config()``.

    Examples
    --------
    >>> from gensbi.core import FlowMatchingMethod
    >>> pipeline = ConditionalPipeline(
    ...     model=my_model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     dim_obs=5, dim_cond=3,
    ...     method=FlowMatchingMethod(),
    ... )
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        method: GenerativeMethod,
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        size=2,
        params=None,
        training_config=None,
    ):
        self.method = method

        # Merge method-specific defaults before super().__init__ which
        # computes derived values from training_config.
        if training_config is None:
            training_config = self.get_default_training_config()
        extra = method.get_extra_training_config()
        for k, v in extra.items():
            training_config.setdefault(k, v)

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        obs_size, cond_size = _normalize_patch_size(size)
        self.obs_ids, self.dim_obs = _resolve_embedding_ids(
            dim_obs, id_embedding_strategy[0], semantic_id=0, size=obs_size
        )
        self.cond_ids, self.dim_cond = _resolve_embedding_ids(
            dim_cond, id_embedding_strategy[1], semantic_id=1, size=cond_size
        )

        self.path = method.build_path(self.training_config, event_shape=(self.dim_obs, self.ch_obs))
        self.loss_obj = method.build_loss(self.path)

    # -- Factory stubs (model-agnostic: user provides model) ----------------

    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic. "
            "Use model-specific pipelines (e.g. Flux1FlowPipeline) for config init."
        )

    def _make_model(self):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic — the user provides the model."
        )

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalPipeline is model-agnostic — the user provides model params."
        )

    # -- Core pipeline methods ----------------------------------------------

    def get_loss_fn(self):
        def loss_fn(model, batch, key):
            obs, cond = batch
            prepared = self.method.prepare_batch(key, obs, self.path)
            model_extras = {
                "cond": cond,
                "obs_ids": self.obs_ids,
                "cond_ids": self.cond_ids,
            }
            return self.loss_obj(model, prepared, model_extras=model_extras)

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)

    def get_sampler(self, x_o, use_ema=True, model_extras=None, **sampler_kwargs):
        """Get a sampler function.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (observed data).
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        model_extras : dict, optional
            Additional keyword arguments passed to the model during sampling
            (e.g. ``{"edge_mask": mask}``). Cannot override the protected keys
            ``cond``, ``obs_ids``, ``cond_ids``.
        **sampler_kwargs
            Forwarded to ``method.build_sampler_fn`` (e.g. ``step_size``,
            ``nsteps``, ``solver``, ``time_grid``).

        Returns
        -------
        Callable
            ``sampler(key, nsamples) -> samples``
        """
        model_wrapped = self.ema_model_wrapped if use_ema else self.model_wrapped

        cond = _expand_dims(x_o)

        _PROTECTED = {"cond", "obs_ids", "cond_ids"}
        extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
        }
        if model_extras:
            conflict = _PROTECTED & model_extras.keys()
            if conflict:
                raise ValueError(
                    f"model_extras cannot override protected keys: {conflict}"
                )
            extras.update(model_extras)

        sampler_fn = self.method.build_sampler_fn(
            model_wrapped, self.path, extras, **sampler_kwargs,
        )

        def sampler(key, nsamples, model_extras=None):
            _extras = model_extras if model_extras is not None else extras
            key, key_init = jax.random.split(key)
            x_init = self.method.sample_init(
                key_init, nsamples,
            )
            return sampler_fn(key, x_init, _extras)

        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : array-like
            Conditioning variable.
        nsamples : int, optional
            Number of samples. Default is 10 000.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """

        x_o_shape = x_o.shape[0] if hasattr(x_o, "shape") else len(x_o)
        if x_o_shape > 1:
            warnings.warn(
                f"x_o has batch dimension {x_o_shape} > 1. "
                "sample() draws all samples for a single condition. "
                "To sample for multiple conditions, use sample_batched() instead.",
                UserWarning,
                stacklevel=2,
            )
        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return sampler(key, nsamples)

    def get_log_prob_fn(self, x_o, use_ema=True, model_extras=None, **kwargs):
        """Get a log-probability function.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (observed data).
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        model_extras : dict, optional
            Additional model extras. Cannot override protected keys.
        **kwargs
            Forwarded to ``method.build_log_prob_fn``.

        Returns
        -------
        Callable
            ``log_prob_fn(x_1) -> log_prob``
        """
        model_wrapped = self.ema_model_wrapped if use_ema else self.model_wrapped

        cond = _expand_dims(x_o)

        _PROTECTED = {"cond", "obs_ids", "cond_ids"}
        extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
        }
        # TODO: this branch is not currently tested, as we don't really ever use it. 
        # Add tests when we find a good usage for this, same below.
        if model_extras:
            conflict = _PROTECTED & model_extras.keys()
            if conflict:
                raise ValueError(
                    f"model_extras cannot override protected keys: {conflict}"
                )
            extras.update(model_extras)

        log_prob_fn = self.method.build_log_prob_fn(
            model_wrapped, self.path, extras, **kwargs,
        )

        def _log_prob(x_1, model_extras=None, *, key=None):
            _extras = model_extras if model_extras is not None else extras
            return log_prob_fn(x_1, _extras, key=key)

        return _log_prob

    def log_prob(self, x_1, x_o, use_ema=True, *, key=None, **kwargs):
        """Compute log-probability of x_1 given x_o.

        Parameters
        ----------
        x_1 : array-like
            Data samples to evaluate.
        x_o : array-like
            Conditioning variable.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        key : jax.random.PRNGKey, optional
            Required when ``exact_divergence=False`` (Hutchinson).
        **kwargs
            Forwarded to :meth:`get_log_prob_fn`.

        Returns
        -------
        Array
            Log-probabilities.
        """
        log_prob_fn = self.get_log_prob_fn(x_o, use_ema=use_ema, **kwargs)
        return log_prob_fn(x_1, key=key)
