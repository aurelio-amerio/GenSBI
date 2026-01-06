import jax.numpy as jnp
import jax
from flax import nnx

from typing import Callable, Tuple, Optional
from jax.numpy import ndarray as Array

from gensbi.flow_matching.loss import ContinuousFMLoss


class ConditionalCFMLoss(ContinuousFMLoss):
    """
    ConditionalCFMLoss is a class that computes the continuous flow matching loss for the Conditional model.

    Parameters
    ----------
        path: Probability path (x-prediction training).
        reduction : str, optional
            Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path, reduction="mean", cfg_scale=None):
        # self.path = path
        # if reduction not in ["None", "mean", "sum"]:
        #     raise ValueError(f"{reduction} is not a valid value for reduction")

        # if reduction == "mean":
        #     self.reduction = jnp.mean
        # elif reduction == "sum":
        #     self.reduction = jnp.sum
        # else:
        #     self.reduction = lambda x: x

        super().__init__(path, reduction)

        self.cfg_scale = cfg_scale

    def __call__(self, vf, batch, cond, obs_ids, cond_ids):
        """
        Evaluates the continuous flow matching loss.

        Parameters
        ----------
            vf : callable
                The vector field model to evaluate.
            batch : tuple
                A tuple containing the input data (x_0, x_1, t).
            cond : jnp.ndarray
                The conditioning data.
            obs_ids : jnp.ndarray
                The observation IDs.
            cond_ids : jnp.ndarray
                The conditioning IDs.

        Returns
        -------
            jnp.ndarray: The computed loss.
        """

        path_sample = self.path.sample(*batch)

        x_t = path_sample.x_t

        if self.cfg_scale is not None:
            key = jax.random.PRNGKey(0)
            conditioned = jax.random.bernoulli(
                key, p=self.cfg_scale, shape=(x_t.shape[0],)
            )
        else:
            conditioned = jnp.ones((x_t.shape[0],), dtype=jnp.bool_)

        model_output = vf(
            t=path_sample.t, obs=x_t, obs_ids=obs_ids, cond=cond, cond_ids=cond_ids, conditioned=conditioned
        )
        loss = model_output - path_sample.dx_t
        loss = jnp.square(loss)

        return self.reduction(loss)


# TODO: WIP
class ConditionalDiffLoss(nnx.Module):
    """
    ConditionalDiffLoss is a class that computes the diffusion score matching loss for the Conditional model.

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
        cond,
        obs_ids,
        cond_ids,
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
            cond : jnp.ndarray
                The conditioning data.
            obs_ids : jnp.ndarray
                The observation IDs.
            cond_ids : jnp.ndarray
                The conditioning IDs.

        Returns
        -------
            Array
                Computed loss.
        """
        x_1, sigma = batch

        path_sample = self.path.sample(key, x_1, sigma)
        batch = path_sample.get_batch()

        # def F_model(x, sigma, obs_ids, cond, cond_ids, **model_extras):
        #     if sigma.ndim == 1:
        #         sigma = sigma[..., None, None]
        #     return model(
        #         t=sigma,
        #         obs=x,
        #         obs_ids=obs_ids,
        #         cond=cond,
        #         cond_ids=cond_ids,
        #         **model_extras,
        #     )

        model_extras = {}
        model_extras["cond"] = cond
        model_extras["obs_ids"] = obs_ids
        model_extras["cond_ids"] = cond_ids

        loss = self.loss_fn(model, batch, loss_mask=None, model_extras=model_extras)

        return loss  # type: ignore
