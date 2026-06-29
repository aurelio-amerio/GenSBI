"""NPE pipeline for discrete normalizing flows (parallel track).

The flow IS the density model: no ``ConditionalWrapper``, no ``GenerativeMethod``.
Trains ``q(obs | cond)`` by max-likelihood. NPE convention: ``obs = theta``,
``cond = x`` (mirrors ``ConditionalPipeline`` so the diagnostics run unchanged).
"""

import warnings

import jax
import jax.numpy as jnp

from gensbi.recipes.pipeline import AbstractPipeline


def _require_channel(x, name="input"):
    """Enforce a tabular channel axis (B, dim, C); reject a bare (B, dim)."""
    x = jnp.asarray(x)
    if x.ndim < 3:
        raise ValueError(
            f"{name} must carry a channel axis (B, dim, C); got shape "
            f"{tuple(x.shape)}. A bare (B, dim) is not accepted — add a trailing "
            f"channel axis (e.g. x[..., None] for C=1).")
    return x


def _warn_if_batched(n):
    """Warn (flow-matching convention) when a single-observation method is given
    a leading batch axis > 1; the caller then proceeds with the first observation."""
    if n > 1:
        warnings.warn(
            f"x_o has batch dimension {n} > 1. sample()/log_prob() use a single "
            "condition. To use multiple conditions, use sample_batched() instead.",
            UserWarning, stacklevel=3,
        )


def _single_obs(x_o):
    """Strip the leading batch axis from ONE observation, keeping the channel
    (and any structured) axes. Warn + take-first on a batch axis > 1."""
    x_o = jnp.asarray(x_o)
    if x_o.ndim < 2:
        raise ValueError(
            "x_o must carry a leading batch axis (e.g. (1, dim_cond, C)); got "
            f"shape {tuple(x_o.shape)}.")
    _warn_if_batched(x_o.shape[0])
    return x_o[0]


def _warn_unused_kwargs(kwargs):
    """Warn that solver-style kwargs are ignored by the (solver-free) flow.

    The flow pipeline mirrors the flow-matching surface (which accepts
    ``**sampler_kwargs``), but a normalizing flow has no ODE/SDE solver, so
    arguments like ``step_size``/``nsteps``/``solver`` do not apply and are
    silently ignored apart from this warning.
    """
    if kwargs:
        keys = ", ".join(sorted(kwargs))
        warnings.warn(
            f"flow pipeline ignores unsupported keyword argument(s): {keys}. "
            "A normalizing flow has no solver, so these have no effect.",
            UserWarning, stacklevel=3,
        )


class ConditionalFlowPipeline(AbstractPipeline):
    """Max-likelihood NPE pipeline wrapping an ``MAFlow``.

    Parameters
    ----------
    model : MAFlow
        A pre-built flow (e.g. ``MAFlow(MAFlowParams(rngs=rngs, dim=dim_obs, cond_dim=dim_cond))``).
    train_dataset, val_dataset : iterable
        Yield ``(obs, cond)`` batches.  Shape is ``(B, dim, C)`` for each
        variable (``C = 1`` for the tabular path; see ``ch_obs``/``ch_cond``).
    dim_obs, dim_cond : int
    ch_obs, ch_cond : int, optional
        Channel count for the obs and cond variables.  Default 1 (tabular SBI).
        Values > 1 enable the ``(B, dim, C)`` channel-passthrough path: the
        channel axis is preserved and forwarded to the flow unchanged (the flow
        must be built with matching ``channels``/``cond_channels`` in
        :class:`~gensbi.models.MAFlowParams`).
    structured_obs, structured_cond : bool, optional
        If ``True``, the modeled variable / condition keeps its native
        structured shape (the model owns it) instead of the tabular
        ``(B, dim, 1)`` layout. Default ``False``.

    Notes
    -----
    Every single-observation method (:meth:`sample`, :meth:`log_prob`,
    :meth:`get_sampler`, :meth:`get_log_prob_fn`) expects ``x_o`` to carry a
    **leading batch axis** (size 1 for one observation) **and a channel axis**:
    shape ``(1, dim_cond, C)`` for tabular, or ``(1,) + per_obs_shape`` for
    structured. A bare ``(B, dim)`` tensor is rejected — add ``[..., None]``
    for ``C = 1``. A batch axis > 1 emits a ``UserWarning`` and the first
    observation is used — pass a batch to :meth:`sample_batched` instead.
    """

    def __init__(self, model, train_dataset, val_dataset, dim_obs, dim_cond,
                 ch_obs=1, ch_cond=1, params=None, training_config=None,
                 structured_obs=False, structured_cond=False):
        super().__init__(
            model, train_dataset, val_dataset, dim_obs, dim_cond,
            ch_obs=ch_obs, ch_cond=ch_cond, params=params,
            training_config=training_config)
        self._standardized = False
        self.structured_obs = structured_obs
        self.structured_cond = structured_cond

    def _prep_obs(self, x):
        x = jnp.asarray(x)
        return x if self.structured_obs else _require_channel(x, "obs")

    def _prep_cond(self, x):
        x = jnp.asarray(x)
        return x if self.structured_cond else _require_channel(x, "cond")

    # --- abstract methods the flow pipeline does not use (mirror ConditionalPipeline) ---
    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        """Not implemented: the flow pipeline requires a pre-built model.

        Raises
        ------
        NotImplementedError
            Always. Construct an ``MAFlow`` and pass it as ``model=`` to the
            pipeline constructor instead.
        """
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built flow; build a `MAFlow` "
            "and pass it as model=.")

    def _make_model(self, params):
        raise NotImplementedError(
            "Pass a pre-built MAFlow as model=; the flow pipeline does not build "
            "models from params.")

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        """Not implemented: the flow pipeline takes a pre-built ``MAFlow``.

        Raises
        ------
        NotImplementedError
            Always. There are no default model params to return; construct an
            ``MAFlow`` directly and pass it as ``model=``.
        """
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built MAFlow; there are no model "
            "params to default.")

    # --- the flow IS the model: no wrapper ---
    def _wrap_model(self):
        self.model_wrapped = self.model
        self.ema_model_wrapped = self.ema_model

    # --- methods implemented in later tasks (Tasks 3, 5, 6) ---
    def get_loss_fn(self):
        """Return the max-likelihood loss function for training.

        Returns a closure ``loss_fn(model, batch, key) -> Array`` that
        computes the mean negative log-likelihood
        ``-mean(log q(obs | cond))``.  ``batch = (obs, cond)`` with each
        element of shape ``(B, dim, 1)``.  NPE convention: ``obs = theta``,
        ``cond = x``.  The ``key`` argument is accepted for interface
        compatibility but is unused.

        Returns
        -------
        loss_fn : Callable
            A function ``(model, batch, key) -> Array`` returning the scalar
            mean negative log-likelihood.
        """
        def loss_fn(model, batch, key):
            obs, cond = batch
            obs = self._prep_obs(obs)
            cond = self._prep_cond(cond)
            return -jnp.mean(model.log_prob(obs, cond))

        return loss_fn

    def fit_standardization(self, obs_data, axis=0):
        """Fit the Standardize bijection buffers from training observations.

        Computes per-dimension mean and standard deviation of ``obs_data``
        and stores them as buffers on both the live model and the EMA model.
        EMA only averages ``Param`` variables, so the non-Param buffers must
        be set explicitly here.  Must be called before :meth:`train` when
        input standardization is desired.

        Parameters
        ----------
        obs_data : Array
            Training observations of shape ``(N, dim_obs)`` or
            ``(N, dim_obs, 1)`` (the autoregressive target; e.g. theta for
            NPE).  For multichannel flows (``ch_obs > 1``) the shape is
            ``(N, dim_obs, C)`` and ``axis=(0, 1)`` yields per-channel stats.
        axis : int or tuple of int, optional
            Reduction axis or axes for the mean/std computation.  Default
            is ``0`` (per-dimension stats over the batch), which is the
            correct choice for the tabular (``C == 1``) path.  Pass
            ``axis=(0, 1)`` for per-channel standardization when ``C > 1``.
        """
        obs = jnp.asarray(obs_data)
        mean = jnp.mean(obs, axis=axis)
        std = jnp.std(obs, axis=axis)
        std = jnp.where(std < 1e-6, 1.0, std)     # guard zero-variance dims
        self.model.set_standardization(mean, std)
        self.ema_model.set_standardization(mean, std)
        self._standardized = True

    def train(self, rngs, nsteps=None, save_model=True):
        """Train the flow model, warning if standardization was skipped.

        Delegates to :meth:`AbstractPipeline.train` after checking that
        :meth:`fit_standardization` was called.

        Parameters
        ----------
        rngs : nnx.Rngs
            Random number generators for training and validation steps.
        nsteps : int or None, optional
            Number of training steps.  If ``None``, taken from
            ``training_config["nsteps"]``.  Default is ``None``.
        save_model : bool, optional
            If ``True`` (default), serialise the model to disk after
            training.

        Returns
        -------
        loss_array : list
            Per-step training losses.
        val_loss_array : list
            Validation losses recorded at each validation checkpoint.
        """
        if not self._standardized:
            warnings.warn(
                "fit_standardization() was not called before train(); the "
                "Standardize bijection stays at identity. Call "
                "pipeline.fit_standardization(theta_train) first if you want "
                "input standardization.",
                UserWarning, stacklevel=2)
        return super().train(rngs, nsteps=nsteps, save_model=save_model)

    def get_sampler(self, x_o, use_ema=True, **kwargs):
        """Return a sampler closure for a single conditioning observation.

        Parameters
        ----------
        x_o : Array
            Single conditioning observation.  Must carry a leading batch axis
            and a channel axis for tabular cond: shape ``(1, dim_cond, C)``.
            For structured cond: ``(1,) + per_observation_shape``.
            A leading batch axis > 1 emits a ``UserWarning`` and the first
            observation is used (use :meth:`sample_batched` for many conditions).
        use_ema : bool, optional
            If ``True`` (default), use the EMA model; otherwise use the
            live model.

        Returns
        -------
        sampler : Callable
            A function ``(key, nsamples) -> Array`` returning the model's
            native output shape ``(nsamples, dim_obs, C)`` (channel always
            carried).
        """
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)                          # (cond_dim, C_cond) or (H,W,C)

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
            return flow.sample(key, cond=cond_b)         # model owns (nsamples, dim, C)
        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True, **kwargs):
        """Draw posterior samples for a single conditioning observation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : Array
            Single conditioning observation carrying a leading batch axis of
            size 1 (see :meth:`get_sampler` for the shape convention). A batch
            axis > 1 warns and the first observation is used.
        nsamples : int, optional
            Number of posterior samples to draw.  Default is 10 000.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(nsamples, dim_obs, 1)`` (or
            ``(nsamples, dim_obs)`` when ``structured_cond=True``).
        """
        return self.get_sampler(x_o, use_ema=use_ema, **kwargs)(key, nsamples)

    def sample_batched(self, key, x_o, nsamples=10_000, *, use_ema=True,
                       **kwargs):
        """Draw posterior samples for a batch of conditioning observations.

        Loops the single-observation sampler over the ``B`` conditions and
        stacks results to ``(nsamples, B, dim_obs, C)`` — the same layout
        as the base pipeline.

        Unlike :class:`~gensbi.recipes.pipeline.AbstractPipeline`
        ``sample_batched`` (which threads the condition through
        ``model_extras``/``obs_ids``/``cond_ids``), this override bakes
        each condition into a :meth:`get_sampler` closure.  Extra
        ``kwargs`` (e.g. ``chunk_size``/``show_progress_bars``) are
        accepted and silently ignored.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key split across the ``B`` conditions.
        x_o : Array
            Batch of observations of shape ``(B, dim_cond, 1)`` or
            ``(B, dim_cond)``.
        nsamples : int, optional
            Number of posterior samples per observation.  Default is
            10 000.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.
        **kwargs : dict, optional
            Extra keyword arguments accepted for interface compatibility
            and silently ignored (e.g. ``chunk_size`` and
            ``show_progress_bars`` from
            :class:`~gensbi.recipes.pipeline.AbstractPipeline`).

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(nsamples, B, dim_obs, 1)`` (or
            ``(nsamples, B, dim_obs)`` when ``structured_cond=True``).
        """
        _warn_unused_kwargs(kwargs)
        x_o = jnp.asarray(x_o)
        B = x_o.shape[0]
        keys = jax.random.split(key, B)
        results = [
            self.get_sampler(x_o[i : i + 1], use_ema=use_ema)(keys[i], nsamples)
            for i in range(B)
        ]
        return jnp.stack(results, axis=1)          # (nsamples, B, dim_obs, 1)

    def get_log_prob_fn(self, x_o, use_ema=True, **kwargs):
        """Return a log-probability closure for a single conditioning observation.

        Parameters
        ----------
        x_o : Array
            Single conditioning observation carrying a leading batch axis of
            size 1 (see :meth:`get_sampler` for the shape convention). A batch
            axis > 1 warns and the first observation is used.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.

        Returns
        -------
        log_prob_fn : Callable
            A function ``(x_1) -> Array`` of shape ``(B,)`` evaluating
            the conditional log-probability ``log q(x_1 | x_o)`` for a
            batch of ``B`` parameter vectors.  ``x_1`` has shape
            ``(B, dim_obs)`` or ``(B, dim_obs, 1)`` on the tabular path,
            or ``(B, dim_obs, C)`` when ``ch_obs > 1``
            (channel-passthrough).
        """
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
            return flow.log_prob(obs, cond_b)            # (B,)
        return log_prob_fn

    def log_prob(self, x_1, x_o, use_ema=True, **kwargs):
        """Evaluate the conditional log-probability for a batch of samples.

        Parameters
        ----------
        x_1 : Array
            Batch of parameter vectors of shape ``(B, dim_obs)`` or
            ``(B, dim_obs, 1)``.
        x_o : Array
            Single conditioning observation carrying a leading batch axis of
            size 1 (see :meth:`get_sampler` for the shape convention). A batch
            axis > 1 warns and the first observation is used.
        use_ema : bool, optional
            If ``True`` (default), use the EMA model.

        Returns
        -------
        log_prob : Array
            Log-probabilities of shape ``(B,)``.
        """
        return self.get_log_prob_fn(x_o, use_ema=use_ema, **kwargs)(x_1)
