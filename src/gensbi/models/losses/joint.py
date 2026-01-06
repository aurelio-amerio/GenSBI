import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Optional
from jax.numpy import ndarray as Array

from gensbi.flow_matching.loss import ContinuousFMLoss


class JointCFMLoss(ContinuousFMLoss):
    """
    JointCFMLoss is a class that computes the continuous flow matching loss for the Joint model.

    Parameters
    ----------
        path: Probability path for training.
        reduction : str
            Reduction method ('none', 'mean', 'sum').
    """

    def __init__(self, path, reduction: str = "mean"):
        super().__init__(path, reduction)

    def __call__(
        self,
        vf: Callable,
        batch: Tuple[Array, Array, Array],
        *args,
        condition_mask: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """
        Evaluate the continuous flow matching loss.

        Parameters
        ----------
            vf : Callable
                Vector field model.
            batch : Tuple[Array, Array, Array]
                Input data (x_0, x_1, t).
            args : Optional[dict]
                Additional arguments.
            condition_mask : Optional[Array]
                Mask for conditioning.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            Array
                Computed loss.
        """
        _, x_1, _ = batch
        path_sample = self.path.sample(*batch)

        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask

        x_t = path_sample.x_t

        if condition_mask is not None:
            # condition_mask = condition_mask.reshape(x_t.shape[0], x_t.shape[1], 1)
            x_t = jnp.where(condition_mask, x_1, x_t)

        model_output = vf(path_sample.t, x_t, *args, **kwargs)

        loss = model_output - path_sample.dx_t
        if condition_mask is not None:
            loss = jnp.where(condition_mask, 0.0, loss)

        return self.reduction(jnp.square(loss))  # type: ignore


class JointDiffLoss(nnx.Module):
    """
    JointDiffLoss is a class that computes the diffusion score matching loss for the Joint model.

    Parameters
    ----------
        path: Probability path for training.
    """

    def __init__(self, path):
        self.path = path

        self.loss_fn = self.path.get_loss_fn()

    def __call__(
        self,
        key: jax.random.PRNGKey,
        model: Callable,
        batch: Tuple[Array, Array, Array],
        condition_mask: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """
        Evaluate the continuous flow matching loss.

        Parameters
        ----------
            key : jax.random.PRNGKey
                Random key for stochastic operations.
            model : Callable
                F model.
            batch : Tuple[Array, Array, Array]
                Input data (x_1, sigma).
            args : Optional[dict]
                Additional arguments.
            condition_mask : Optional[Array]
                Mask for conditioning.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            Array
                Computed loss.
        """
        x_1, sigma = batch

        path_sample = self.path.sample(key, x_1, sigma)
        batch = path_sample.get_batch()

        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask

        loss = self.loss_fn(model, batch, loss_mask=condition_mask, model_extras=kwargs)

        return loss  # type: ignore
