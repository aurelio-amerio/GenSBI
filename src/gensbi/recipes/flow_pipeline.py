"""NPE pipeline for discrete normalizing flows (parallel track).

The flow IS the density model: no ``ConditionalWrapper``, no ``GenerativeMethod``.
Trains ``q(obs | cond)`` by max-likelihood. NPE convention: ``obs = theta``,
``cond = x`` (mirrors ``ConditionalPipeline`` so the diagnostics run unchanged).
"""

import warnings

import jax
import jax.numpy as jnp

from gensbi.recipes.pipeline import AbstractPipeline
from gensbi.utils.math import _expand_dims


def _squeeze_ch(x):
    """``(B, dim, 1) -> (B, dim)``; pass ``(B, dim)`` through. Asserts ch == 1."""
    x = jnp.asarray(x)
    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(
                f"flow pipeline requires a singleton channel axis (ch == 1), "
                f"got shape {x.shape}")
        return x[..., 0]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected (B, dim) or (B, dim, 1), got shape {tuple(x.shape)}")


def _warn_if_batched(n):
    """Warn (flow-matching convention) when a single-observation method is given
    a leading batch axis > 1; the caller then proceeds with the first observation."""
    if n > 1:
        warnings.warn(
            f"x_o has batch dimension {n} > 1. sample()/log_prob() use a single "
            "condition. To use multiple conditions, use sample_batched() instead.",
            UserWarning, stacklevel=3,
        )


def _single_cond(x_o):
    """Reduce a single conditioning observation to a 1-D ``(dim_cond,)`` vector.

    Mirrors the flow-matching pipeline convention: ``x_o`` carries a leading
    batch axis (size 1 for one observation); a batch axis > 1 warns and the
    first observation is used.
    """
    x_o = jnp.asarray(x_o)
    if x_o.ndim >= 2:
        _warn_if_batched(x_o.shape[0])
        x_o = x_o[0]                       # take the first observation
    x_o = jnp.squeeze(x_o)
    if x_o.ndim == 0:
        x_o = x_o[None]
    if x_o.ndim != 1:
        raise ValueError(
            f"x_o must reduce to a single (dim_cond,) vector; got shape "
            f"{tuple(jnp.asarray(x_o).shape)}. sample()/log_prob() take ONE "
            f"observation at a time.")
    return x_o


def _structured_cond(x_o):
    """Strip the leading batch axis from a single structured observation.

    For ``structured_cond=True`` the conditioner owns the per-observation
    shape, so ``x_o`` carries a leading batch axis (size 1 for one observation,
    as produced by e.g. ``x_o[i:i+1]``). That axis is removed unconditionally —
    never by a ``shape[0] == 1`` heuristic — so a genuine size-1 *data* axis
    (e.g. a ``(1, 1, W, C)`` image with ``H == 1``) is never mistaken for the
    batch axis and silently dropped. Mirroring the flow-matching pipeline, a
    batch axis > 1 warns and the first observation is used.
    """
    cond = jnp.asarray(x_o)
    if cond.ndim < 1:
        raise ValueError(
            "structured_cond x_o must carry a leading batch axis (e.g. shape "
            f"(1,) + per_observation_shape); got a scalar of shape "
            f"{tuple(cond.shape)}.")
    _warn_if_batched(cond.shape[0])
    return cond[0]                        # take the first observation


class ConditionalFlowPipeline(AbstractPipeline):
    """Max-likelihood NPE pipeline wrapping an ``MAFlow``.

    Parameters
    ----------
    model : MAFlow
        A pre-built flow (e.g. ``MAFlow(MAFlowParams(rngs=rngs, dim=dim_obs, cond_dim=dim_cond))``).
    train_dataset, val_dataset : iterable
        Yield ``(obs, cond)`` batches of shape ``(B, dim, 1)`` each.
    dim_obs, dim_cond : int
    ch_obs, ch_cond : int
        Must both be 1 (tabular SBI). Default 1.
    structured_obs, structured_cond : bool, optional
        If ``True``, the modeled variable / condition keeps its native
        structured shape (the model owns it) instead of the tabular
        ``(B, dim, 1)`` layout. Default ``False``.

    Notes
    -----
    Following the flow-matching pipelines, every single-observation method
    (:meth:`sample`, :meth:`log_prob`, :meth:`get_sampler`,
    :meth:`get_log_prob_fn`) expects ``x_o`` to carry a **leading batch axis**
    (size 1 for one observation; tabular ``x_o`` may also be a bare
    ``(dim_cond,)`` vector). A batch axis > 1 emits a ``UserWarning`` and the
    first observation is used — pass a batch of conditions to
    :meth:`sample_batched` instead.
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
        return x if self.structured_obs else _squeeze_ch(x)

    def _prep_cond(self, x):
        return x if self.structured_cond else _squeeze_ch(x)

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

    def fit_standardization(self, obs_data):
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
            NPE).
        """
        obs = jnp.asarray(obs_data)
        if not self.structured_obs and obs.ndim == 3:
            obs = _squeeze_ch(obs)
        mean = jnp.mean(obs, axis=0)
        std = jnp.std(obs, axis=0)
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

    def get_sampler(self, x_o, use_ema=True):
        """Return a sampler closure for a single conditioning observation.

        When ``structured_cond=True``, the returned sampler produces
        ``(nsamples, dim_obs)`` instead of ``(nsamples, dim_obs, 1)``
        because the model's conditioner owns the shape (no
        ``_expand_dims``).

        Parameters
        ----------
        x_o : Array
            Single observation used as the conditioning input.  Shape
            ``(dim_cond,)`` or ``(1, dim_cond)`` for the tabular path.  When
            ``structured_cond=True`` the conditioner owns the per-observation
            shape and ``x_o`` must carry a leading batch axis of size 1, i.e.
            ``(1,) + per_observation_shape`` (e.g. ``(1, H, W, C)``); that
            batch axis is stripped, so a genuine size-1 data axis is preserved.
            A leading batch axis > 1 emits a ``UserWarning`` and the first
            observation is used (use :meth:`sample_batched` for many conditions).
        use_ema : bool, optional
            If ``True`` (default), use the EMA model; otherwise use the
            live model.

        Returns
        -------
        sampler : Callable
            A function ``(key, nsamples) -> Array`` of shape
            ``(nsamples, dim_obs, 1)`` (or ``(nsamples, dim_obs)`` when
            ``structured_cond=True``).
        """
        flow = self.ema_model if use_ema else self.model
        if self.structured_cond:
            cond = _structured_cond(x_o)             # strip the leading batch axis

            def sampler(key, nsamples):
                cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
                return flow.sample(key, cond=cond_b)  # (nsamples, dim_obs)
            return sampler

        cond = _single_cond(x_o)                      # (dim_cond,)  [v1 path]

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples, cond.shape[0]))
            samples = flow.sample(key, cond=cond_b)    # (nsamples, dim_obs)
            return _expand_dims(samples)               # (nsamples, dim_obs, 1)

        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True):
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
        return self.get_sampler(x_o, use_ema=use_ema)(key, nsamples)

    def sample_batched(self, key, x_o, nsamples=10_000, *, use_ema=True,
                       **kwargs):
        """Draw posterior samples for a batch of conditioning observations.

        Loops the single-observation sampler over the ``B`` conditions and
        stacks results to ``(nsamples, B, dim_obs, 1)`` — the same layout
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
        x_o = jnp.asarray(x_o)
        B = x_o.shape[0]
        keys = jax.random.split(key, B)
        results = [
            self.get_sampler(x_o[i : i + 1], use_ema=use_ema)(keys[i], nsamples)
            for i in range(B)
        ]
        return jnp.stack(results, axis=1)          # (nsamples, B, dim_obs, 1)

    def get_log_prob_fn(self, x_o, use_ema=True):
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
            batch of ``B`` parameter vectors.
        """
        flow = self.ema_model if use_ema else self.model
        if self.structured_cond:
            cond = _structured_cond(x_o)             # strip the leading batch axis

            def log_prob_fn(x_1):
                obs = self._prep_obs(x_1)
                cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
                return flow.log_prob(obs, cond_b)
            return log_prob_fn

        cond = _single_cond(x_o)                  # (dim_cond,)  [v1 path]

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)              # (B, dim_obs)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0], cond.shape[0]))
            return flow.log_prob(obs, cond_b)      # (B,)

        return log_prob_fn

    def log_prob(self, x_1, x_o, use_ema=True):
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
        return self.get_log_prob_fn(x_o, use_ema=use_ema)(x_1)
