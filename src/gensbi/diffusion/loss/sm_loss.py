"""
Score matching loss with unified interface.
"""

import jax.numpy as jnp


class SMLoss:
    """Score matching loss with a uniform ``(model, batch, ...)`` interface.

    Wraps ``SMPath.get_loss_fn()`` so that the calling convention matches
    :class:`~gensbi.flow_matching.loss.FMLoss` and
    :class:`~gensbi.diffusion.loss.EDMLoss`.

    Parameters
    ----------
    path : SMPath
        The score matching path.
    weights : Array, optional
        Weights for the loss, applied element-wise before reduction.
    """

    def __init__(self, path, weights=None):
        self.path = path
        self.loss_fn = path.get_loss_fn()
        self.weights = jnp.asarray(weights) if weights is not None else None

    def __call__(self, model, batch, condition_mask=None, model_extras=None):
        """Evaluate the score matching loss.

        Parameters
        ----------
        model : Callable
            The score model.
        batch : tuple
            ``(x_0, x_1, t)`` — standard normal noise, clean data,
            and diffusion time.
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
        loss_batch = path_sample.get_batch()

        if self.weights is not None:
            weights = jnp.broadcast_to(self.weights, x_1.shape)
        else:
            weights = None

        return self.loss_fn(
            model, loss_batch,
            condition_mask=condition_mask,
            weights=weights,
            model_extras=model_extras,
        )
