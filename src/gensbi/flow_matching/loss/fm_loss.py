"""
Flow matching loss with unified interface.

This module provides :class:`FMLoss`, which computes the continuous flow
matching loss using an :class:`AffineProbPath` and calls the model with
named arguments matching the GenSBI wrapper convention.
"""

import jax.numpy as jnp
from jax import Array


class FMLoss:
    """Flow matching loss with a uniform ``(model, batch, ...)`` interface.

    Computes the path sample from ``(x_0, x_1, t)`` and calls the model
    with named arguments ``(obs=x_t, t=t, **model_extras)`` so that the
    argument order matches ``ConditionalWrapper``, ``JointWrapper``, and
    ``UnconditionalWrapper``.

    Parameters
    ----------
    path : AffineProbPath
        The probability path.
    """

    def __init__(self, path):
        self.path = path
        self.reduction = jnp.mean

    def __call__(self, model, batch, condition_mask=None, model_extras=None):
        """Evaluate the flow matching loss.

        Parameters
        ----------
        model : Callable
            The velocity field model.
        batch : tuple
            ``(x_0, x_1, t)`` — source noise, target data, and time.
        condition_mask : Array, optional
            Conditioning mask (for joint models).
        model_extras : dict, optional
            Additional model keyword arguments.

        Returns
        -------
        Array
            Scalar loss.
        """
        if model_extras is None:
            model_extras = {}

        x_0, x_1, t = batch
        path_sample = self.path.sample(x_0, x_1, t)

        x_t = path_sample.x_t

        if condition_mask is not None:
            condition_mask_broad = jnp.broadcast_to(condition_mask, x_1.shape)
            x_t = jnp.where(condition_mask_broad, x_1, x_t)
            model_extras["condition_mask"] = condition_mask

        model_output = model(obs=x_t, t=path_sample.t, **model_extras)

        loss = jnp.square(model_output - path_sample.dx_t)

        if condition_mask is not None:
            loss = jnp.where(condition_mask_broad, 0.0, loss)

        return self.reduction(loss)
