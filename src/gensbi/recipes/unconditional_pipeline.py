"""
Pipeline for training and using a Unconditional model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from numpyro import distributions as dist


from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver, BaseFmSDESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler
from gensbi.diffusion.solver import SMSolver, SMPFSolver

from gensbi.models import (
    UnconditionalCFMLoss,
    UnconditionalWrapper,
    UnconditionalEDMLoss,
)
from gensbi.models.losses import UnconditionalSMLoss

from gensbi.recipes.utils import init_ids_1d, build_edm_path, build_sm_path

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

from gensbi.recipes.pipeline import AbstractPipeline


# ---------------------------------------------------------------------------
# Deprecated pipeline classes (replaced by UnconditionalPipeline)
# ---------------------------------------------------------------------------


class _DeprecatedUnconditionalPipeline:
    """Base for deprecated unconditional pipeline stubs."""

    _message = ""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(self._message)

    @classmethod
    def get_default_training_config(cls, **kwargs):
        raise RuntimeError(cls._message)


class UnconditionalFlowPipeline(_DeprecatedUnconditionalPipeline):
    _message = (
        "UnconditionalFlowPipeline has been removed. "
        "Use UnconditionalPipeline(method=FlowMatchingMethod(), ...) instead."
    )


class UnconditionalDiffusionPipeline(_DeprecatedUnconditionalPipeline):
    _message = (
        "UnconditionalDiffusionPipeline has been removed. "
        "Use UnconditionalPipeline(method=DiffusionEDMMethod(), ...) instead."
    )


class UnconditionalSMPipeline(_DeprecatedUnconditionalPipeline):
    _message = (
        "UnconditionalSMPipeline has been removed. "
        "Use UnconditionalPipeline(method=ScoreMatchingMethod(), ...) instead."
    )


# ---------------------------------------------------------------------------
# Unified UnconditionalPipeline (Phase 2)
# ---------------------------------------------------------------------------

from gensbi.core.generative_method import GenerativeMethod


class UnconditionalPipeline(AbstractPipeline):
    """Model-agnostic unconditional pipeline parameterized by a ``GenerativeMethod``.

    Unlike the method-specific classes above (``UnconditionalFlowPipeline``,
    ``UnconditionalDiffusionPipeline``, ``UnconditionalSMPipeline``), this
    class works with **any** generative method and **any** user-provided
    model that conforms to the ``UnconditionalWrapper`` interface.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : iterable
        Training dataset yielding ``x_1`` batches (not tuples).
    val_dataset : iterable
        Validation dataset.
    dim_obs : int
        Dimension of the data space.
    method : GenerativeMethod
        Strategy object (e.g. ``FlowMatchingMethod()``,
        ``DiffusionEDMMethod()``, ``ScoreMatchingMethod()``).
    ch_obs : int, optional
        Number of channels per token. Default is 1.
    params : optional
        Model parameters (stored but not used directly).
    training_config : dict, optional
        Training configuration.

    Examples
    --------
    >>> from gensbi.core import FlowMatchingMethod
    >>> pipeline = UnconditionalPipeline(
    ...     model=my_model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     dim_obs=9,
    ...     method=FlowMatchingMethod(),
    ... )
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        method: GenerativeMethod,
        ch_obs=1,
        params=None,
        training_config=None,
    ):
        self.method = method

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
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids, self.dim_obs = init_ids_1d(self.dim_obs)

        self.path = method.build_path(self.training_config)
        self.loss_obj = method.build_loss(self.path)

    # -- Factory stubs ------------------------------------------------------

    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "UnconditionalPipeline is model-agnostic. "
            "Use model-specific pipelines for config init."
        )

    def _make_model(self):
        raise NotImplementedError(
            "UnconditionalPipeline is model-agnostic — the user provides the model."
        )

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "UnconditionalPipeline is model-agnostic — the user provides model params."
        )

    # -- Core pipeline methods ----------------------------------------------

    def get_loss_fn(self):
        def loss_fn(model, batch, key):
            x_1 = batch
            prepared = self.method.prepare_batch(key, x_1, self.path)
            model_extras = {"obs_ids": self.obs_ids}
            return self.loss_obj(
                model, prepared,
                condition_mask=None,
                model_extras=model_extras,
            )

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)

    def get_sampler(self, use_ema=True, **sampler_kwargs):
        """Get a sampler function.

        Parameters
        ----------
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        **sampler_kwargs
            Forwarded to ``method.build_sampler_fn``.

        Returns
        -------
        Callable
            ``sampler(key, nsamples) -> samples``
        """
        model_wrapped = self.ema_model_wrapped if use_ema else self.model_wrapped

        model_extras = {"obs_ids": self.obs_ids}

        sampler_fn = self.method.build_sampler_fn(
            model_wrapped, self.path, model_extras, **sampler_kwargs,
        )

        def sampler(key, nsamples):
            key, key_init = jax.random.split(key)
            x_init = self.method.sample_init(
                key_init, (nsamples, self.dim_obs, self.ch_obs), self.path,
            )
            return sampler_fn(key, x_init)

        return sampler

    def sample(self, key, nsamples=10_000, use_ema=True, **sampler_kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
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
        sampler = self.get_sampler(use_ema=use_ema, **sampler_kwargs)
        return sampler(key, nsamples)

    def sample_batched(self, *args, **kwargs):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalPipeline."
        )
