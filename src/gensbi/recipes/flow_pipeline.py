"""NPE pipeline for discrete normalizing flows (parallel track).

The flow IS the density model: no ``ConditionalWrapper``, no ``GenerativeMethod``.
Trains ``q(obs | cond)`` by max-likelihood. NPE convention: ``obs = theta``,
``cond = x`` (mirrors ``ConditionalPipeline`` so the diagnostics run unchanged).
"""

import warnings

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


def _single_cond(x_o):
    """Reduce a single conditioning observation to a 1-D ``(dim_cond,)`` vector."""
    x_o = jnp.squeeze(jnp.asarray(x_o))
    if x_o.ndim == 0:
        x_o = x_o[None]
    if x_o.ndim != 1:
        raise ValueError(
            f"x_o must reduce to a single (dim_cond,) vector; got shape "
            f"{tuple(jnp.asarray(x_o).shape)}. sample()/log_prob() take ONE "
            f"observation at a time.")
    return x_o


class ConditionalFlowPipeline(AbstractPipeline):
    """Max-likelihood NPE pipeline wrapping a Phase-0 ``Flow``.

    Parameters
    ----------
    model : Flow
        A pre-built flow (e.g. ``make_maf(rngs, dim=dim_obs, cond_dim=dim_cond)``).
    train_dataset, val_dataset : iterable
        Yield ``(obs, cond)`` batches of shape ``(B, dim, 1)`` each.
    dim_obs, dim_cond : int
    ch_obs, ch_cond : int
        Must both be 1 (tabular SBI). Default 1.
    """

    def __init__(self, model, train_dataset, val_dataset, dim_obs, dim_cond,
                 ch_obs=1, ch_cond=1, params=None, training_config=None):
        super().__init__(
            model, train_dataset, val_dataset, dim_obs, dim_cond,
            ch_obs=ch_obs, ch_cond=ch_cond, params=params,
            training_config=training_config)
        self._standardized = False

    # --- abstract methods the flow pipeline does not use (mirror ConditionalPipeline) ---
    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built Flow; build it with "
            "make_maf and pass it as model=.")

    def _make_model(self, params):
        raise NotImplementedError(
            "Pass a pre-built Flow as model=; the flow pipeline does not build "
            "models from params.")

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built Flow; there are no model "
            "params to default.")

    # --- the flow IS the model: no wrapper ---
    def _wrap_model(self):
        self.model_wrapped = self.model
        self.ema_model_wrapped = self.ema_model

    # --- methods implemented in later tasks (Tasks 3, 5, 6) ---
    def get_loss_fn(self):
        raise NotImplementedError("Implemented in Task 3.")

    def get_sampler(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 5.")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 5.")

    def get_log_prob_fn(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 6.")

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 6.")
