"""
Pipeline for training and using a Joint model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp

from gensbi.recipes.utils import init_ids_joint, build_edm_path, build_sm_path

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler
from gensbi.diffusion.solver import EDMSolver

from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler

from gensbi.core.prior import make_gaussian_prior


from einops import repeat

from gensbi.models import JointWrapper

import numpyro.distributions as dist

from gensbi.utils.model_wrapping import _expand_dims

import os
import yaml

from gensbi.recipes.pipeline import AbstractPipeline, ModelEMA, _chunked_draw, _sample_concat_axis

import warnings


def sample_structured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    """
    Sample structured conditional masks for the Joint model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for sampling.
    num_samples : int
        Number of samples to generate.
    theta_dim : int
        Dimension of the parameter space.
    x_dim : int
        Dimension of the observation space.
    p_joint : float
        Probability of selecting the joint mask.
    p_posterior : float
        Probability of selecting the posterior mask.
    p_likelihood : float
        Probability of selecting the likelihood mask.
    p_rnd1 : float
        Probability of selecting the first random mask.
    p_rnd2 : float
        Probability of selecting the second random mask.
    rnd1_prob : float
        Probability of a True value in the first random mask.
    rnd2_prob : float
        Probability of a True value in the second random mask.

    Returns
    -------
    condition_mask : jnp.ndarray
        Array of shape (num_samples, theta_dim + x_dim) with boolean masks.

    """
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    joint_mask = jnp.array([False] * (theta_dim + x_dim), dtype=jnp.bool_)
    posterior_mask = jnp.array([False] * theta_dim + [True] * x_dim, dtype=jnp.bool_)
    likelihood_mask = jnp.array([True] * theta_dim + [False] * x_dim, dtype=jnp.bool_)
    random1_mask = jax.random.bernoulli(
        key2, rnd1_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    random2_mask = jax.random.bernoulli(
        key3, rnd2_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    mask_options = jnp.stack(
        [joint_mask, posterior_mask, likelihood_mask, random1_mask, random2_mask],
        axis=0,
    )  # (5, theta_dim + x_dim)
    idx = jax.random.choice(
        key1,
        5,
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
    )
    condition_mask = mask_options[idx]
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask[..., None]


def sample_condition_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    kind="structured",
):

    if kind == "structured":
        condition_mask = sample_structured_conditional_mask(
            key,
            num_samples,
            theta_dim,
            x_dim,
        )
    elif kind == "posterior":
        condition_mask = jnp.array(
            [False] * theta_dim + [True] * x_dim, dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    elif kind == "likelihood":
        condition_mask = jnp.array(
            [True] * theta_dim + [False] * x_dim, dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    elif kind == "joint":
        condition_mask = jnp.array(
            [False] * (theta_dim + x_dim), dtype=jnp.bool_
        ).reshape(1, -1, 1)
        condition_mask = jnp.broadcast_to(
            condition_mask, (num_samples, theta_dim + x_dim, 1)
        )
    else:
        raise ValueError(f"Unknown kind {kind} for condition mask.")

    return condition_mask



# ---------------------------------------------------------------------------

from gensbi.core.generative_method import GenerativeMethod


class JointPipeline(AbstractPipeline):
    """Model-agnostic joint pipeline parameterized by a ``GenerativeMethod``.

    Unlike the old method-specific pipeline classes, this class works with
    **any** generative method and **any** user-provided model that conforms
    to the ``JointWrapper`` interface.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : iterable
        Training dataset yielding concatenated ``x_1`` batches
        (obs and cond concatenated along the token dimension).
    val_dataset : iterable
        Validation dataset.
    dim_obs : int
        Dimension of the observation/parameter space.
    dim_cond : int
        Dimension of the conditioning space.
    method : GenerativeMethod
        Strategy object (e.g. ``FlowMatchingMethod()``,
        ``DiffusionEDMMethod()``, ``ScoreMatchingMethod()``).
    ch_obs : int, optional
        Number of channels per token. Default is 1.
    condition_mask_kind : str, optional
        Kind of condition mask. One of ``"structured"`` or ``"posterior"``.
        Default is ``"structured"``.
    params : optional
        Model parameters (stored but not used directly).
    training_config : dict, optional
        Training configuration.
    mesh : jax.sharding.Mesh or None, optional
        Device mesh for multi-device training (also ``"auto"`` or a dict of
        :func:`gensbi.sharding.make_mesh` kwargs). If None, falls back to
        ``training_config["mesh"]``; absent too, training stays on one device.
    sharding_rule : {"replicated", "fsdp", "tensor"} or None, optional
        Logical-to-mesh axis rule used when a mesh is set. If None, falls back
        to ``training_config["sharding_rule"]``, then ``"replicated"``.
        ``"fsdp"`` and ``"tensor"`` are experimental.

    Examples
    --------
    >>> from gensbi.core import FlowMatchingMethod
    >>> pipeline = JointPipeline(
    ...     model=my_model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     dim_obs=2, dim_cond=7,
    ...     method=FlowMatchingMethod(),
    ... )
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        method: GenerativeMethod,
        ch_obs=1,
        condition_mask_kind="structured",
        params=None,
        training_config=None,
        mesh=None,
        sharding_rule=None,
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
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
            mesh=mesh,
            sharding_rule=sharding_rule,
        )

        self.dim_joint = self.dim_obs + self.dim_cond

        self.node_ids, self.obs_ids, self.cond_ids = init_ids_joint(
            self.dim_obs, self.dim_cond
        )

        self.path = method.build_path(self.training_config, event_shape=(self.dim_joint, self.ch_obs))

        # OneFlow reweighting (Eq. 4, https://arxiv.org/pdf/2601.22951v1), currently disabled
        # upweight parameter (obs) dimensions by d_cond / d_obs to balance
        # gradient magnitudes when d_cond >> d_obs.
        # loss_weights = jnp.ones(dim_obs + dim_cond)
        # loss_weights = loss_weights.at[jnp.arange(dim_obs)].set(dim_cond / dim_obs)
        # # reshape to (1, dim_joint, 1) to broadcast over (batch, dim_joint, ch)
        # loss_weights = loss_weights.reshape(1, -1, 1)
        loss_weights = None
        self.loss_obj = method.build_loss(self.path, weights=loss_weights)

        if self.dim_cond == 0:
            raise ValueError(
                "JointPipeline initialized with dim_cond=0. "
                "Use UnconditionalPipeline instead."
            )

        if condition_mask_kind not in ("structured", "posterior"):
            raise ValueError(
                f"condition_mask_kind must be one of ['structured', 'posterior'], "
                f"got {condition_mask_kind}."
            )
        self.condition_mask_kind = condition_mask_kind

    # -- Factory stubs ------------------------------------------------------

    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "JointPipeline is model-agnostic. "
            "Use model-specific pipelines for config init."
        )

    def _make_model(self):
        raise NotImplementedError(
            "JointPipeline is model-agnostic — the user provides the model."
        )

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "JointPipeline is model-agnostic — the user provides model params."
        )

    # -- Core pipeline methods ----------------------------------------------

    def get_loss_fn(self):
        def loss_fn(model, x_1, key):
            batch_size = x_1.shape[0]
            rng_batch, rng_condition = jax.random.split(key)

            prepared = self.method.prepare_batch(rng_batch, x_1, self.path)

            condition_mask = sample_condition_mask(
                rng_condition,
                batch_size,
                self.dim_obs,
                self.dim_cond,
                kind=self.condition_mask_kind,
            )

            model_extras = {
                "node_ids": self.node_ids,
                "condition_mask": condition_mask,
            }
            return self.loss_obj(
                model,
                prepared,
                condition_mask=condition_mask,
                model_extras=model_extras,
            )

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)

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
            Forwarded to ``method.build_sampler_fn``.

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
        # With a mesh, the extras (cond, ids, ...) are replicated over it.
        extras = self._replicate(extras)

        sampler_fn = self.method.build_sampler_fn(
            model_wrapped,
            self.path,
            extras,
            **sampler_kwargs,
        )

        axis = _sample_concat_axis(sampler_kwargs)

        def sampler(key, nsamples, model_extras=None):
            _extras = self._replicate(model_extras) if model_extras is not None else extras
            key, key_init = jax.random.split(key)
            # Marginalize joint prior to obs dims. This matches
            # JointWrapper.conditioned() which assumes obs = first dim_obs
            # dims, cond = remaining dims.
            # When supporting arbitrary condition masks, update both this
            # and the wrapper.
            # With a mesh, x_init is padded to the data-parallel size and
            # sharded over it; the extras (cond, ids) are replicated.
            x_init, n_pad = self._sharded_init(
                lambda k, n: self.method.sample_init(k, n)[:, :self.dim_obs, :],
                key_init, nsamples,
            )
            cond_i = _extras.get("cond")
            if n_pad != nsamples and cond_i is not None and cond_i.shape[0] == nsamples > 1:
                # x_o with one condition per sample: pad it like x_init by
                # repeating the last row (the padded outputs are trimmed).
                cond_i = jnp.concatenate([cond_i, jnp.repeat(cond_i[-1:], n_pad - nsamples, axis=0)])
                _extras = {**_extras, "cond": self._replicate(cond_i)}
            return self._trim(sampler_fn(key, x_init, _extras), nsamples, axis)

        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True,
               chunk_size=None, show_progress_bars=True, **sampler_kwargs):
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
        chunk_size : int, optional
            Maximum number of samples drawn per device call. ``None``
            (default) draws everything in one call — identical to the
            historical behavior.
        show_progress_bars : bool, optional
            Show a progress bar over chunks (only when chunking is
            active). Default is True.
        **sampler_kwargs
            Forwarded to :meth:`get_sampler`.

        Returns
        -------
        Array
            Samples of shape ``(nsamples, dim_obs, ch_obs)``.
        """

        # Warn on the canonical (batch, dim, ch) shape: a bare 1-D
        # observation ``(dim_cond,)`` is a single condition, not a batch.
        x_o_shape = _expand_dims(jnp.asarray(x_o)).shape[0]
        if x_o_shape > 1:
            warnings.warn(
                f"x_o has batch dimension {x_o_shape} > 1. "
                "sample() draws all samples for a single condition. "
                "To sample for multiple conditions, use sample_batched() instead.",
                UserWarning,
                stacklevel=2,
            )
        sampler = self.get_sampler(x_o, use_ema=use_ema, **sampler_kwargs)
        return _chunked_draw(
            sampler, key, nsamples, chunk_size,
            show_progress_bars=show_progress_bars,
            concat_axis=_sample_concat_axis(sampler_kwargs),
        )

    def get_log_prob_fn(self, x_o, use_ema=True, prior=None, model_extras=None, **kwargs):
        """Get a log-probability function.

        Parameters
        ----------
        x_o : array-like
            Conditioning variable (observed data).
        use_ema : bool, optional
            Whether to use the EMA model. Default is True.
        prior : numpyro.distributions.Distribution, optional
            Obs-space prior for log-probability evaluation. The method's
            prior lives on the full joint space ``(dim_joint, ch)`` and
            cannot be automatically marginalized for arbitrary priors.

            - **Default Gaussian**: auto-constructed — no need to provide.
            - **Custom prior**: must supply the correct obs-space marginal.
        model_extras : dict, optional
            Additional model extras. Cannot override protected keys.
        **kwargs
            Forwarded to ``method.build_log_prob_fn``.

        Returns
        -------
        Callable
            ``log_prob_fn(x_1) -> log_prob``

        Raises
        ------
        ValueError
            If the joint prior is non-Gaussian and no ``prior`` is provided.
        """

        if prior is not None:
            log_p0 = prior.log_prob
        elif not self.method.has_custom_prior:
            # Auto-constructed default prior — we know the marginal is a
            # standard Gaussian on the obs dims.
            obs_prior = make_gaussian_prior(self.dim_obs, self.ch_obs)
            log_p0 = obs_prior.log_prob
        else:
            raise ValueError(
                "Joint pipeline with a custom prior requires an explicit "
                "`prior` for log_prob — the obs-space marginal of the joint "
                "prior. Pass a numpyro distribution whose event_shape "
                "matches (dim_obs, ch_obs)."
            )

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
        # Broadcast to every row: replicated over the mesh (no-op without one).
        extras = self._replicate(extras)

        log_prob_fn = self.method.build_log_prob_fn(
            model_wrapped,
            self.path,
            extras,
            log_prior=log_p0,
            **kwargs,
        )

        def _log_prob(x_1, model_extras=None, *, key=None):
            _extras = self._replicate(model_extras) if model_extras is not None else extras
            return log_prob_fn(
                self._place_batch_or_replicate(x_1), _extras, key=self._replicate(key)
            )

        return _log_prob

    def log_prob(self, x_1, x_o, use_ema=True, prior=None, *, key=None, **kwargs):
        """Compute log-probability of x_1 given x_o.

        Parameters
        ----------
        x_1 : array-like
            Data samples to evaluate.
        x_o : array-like
            Conditioning variable.
        use_ema : bool, optional
            Use the EMA model. Default is True.
        prior : numpyro.distributions.Distribution, optional
            Obs-space prior distribution.
            See :meth:`get_log_prob_fn` for details.
        key : jax.random.PRNGKey, optional
            Required when ``exact_divergence=False`` (Hutchinson).
        **kwargs
            Forwarded to :meth:`get_log_prob_fn`.

        Returns
        -------
        Array
            Log-probabilities.
        """
        log_prob_fn = self.get_log_prob_fn(
            x_o, use_ema=use_ema, prior=prior, **kwargs
        )
        return log_prob_fn(x_1, key=key)
