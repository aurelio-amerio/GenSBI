"""
Standard score matching probability path implementation.

This module implements the probability path for standard score matching diffusion
models, supporting VP and VE SDE formulations.

Based on "Score-Based Generative Modeling through Stochastic Differential Equations"
by Song et al., 2021. https://arxiv.org/abs/2011.13456
"""

import jax
from jax import Array
from jax import numpy as jnp
from typing import Callable, Any

from gensbi.diffusion.path.path_sample import SMPathSample


class SMPath:
    """
    Score Matching probability path.

    This class constructs noised samples for standard score matching training
    using the forward SDE's marginal distributions.

    The noising is: x_t = mean_coeff(t) * x_1 + std(t) * epsilon

    Parameters
    ----------
        sde: The SDE scheduler object (VPSmScheduler or VESmScheduler).

    Example:
        .. code-block:: python

            from gensbi.diffusion.path.sm_path import SMPath
            from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler
            import jax, jax.numpy as jnp
            sde = VPSmScheduler()
            path = SMPath(sde)
            key = jax.random.PRNGKey(0)
            x_1 = jax.random.normal(key, (32, 2))
            sample = path.sample(key, x_1)
            print(sample.x_t.shape)
            # (32, 2)
    """

    def __init__(self, sde) -> None:
        """
        Initialize the SMPath with an SDE scheduler.

        Parameters
        ----------
            sde: The SDE scheduler object.

        Raises
        ------
            AssertionError
                If sde name is not one of 'SM-VP' or 'SM-VE'.
        """
        self.sde = sde
        assert self.sde.name in [
            "SM-VP",
            "SM-VE",
        ], f"SDE must be one of ['SM-VP', 'SM-VE'], got {self.sde.name}."
        return

    @property
    def name(self) -> str:
        """Returns the name of the SDE."""
        return self.sde.name

    def sample(self, key: Array, x_1: Array) -> SMPathSample:
        r"""
        Sample from the score matching probability path.

        Constructs x_t = mean_coeff(t) * x_1 + std(t) * epsilon.

        Parameters
        ----------
            key : Array
                JAX random key.
            x_1 : Array
                Target data point, shape (batch_size, ...).

        Returns
        -------
            SMPathSample
                A sample from the SM path.
        """
        key1, key2 = jax.random.split(key)

        # Sample time
        t_shape = (x_1.shape[0],) + (1,) * (x_1.ndim - 1)
        t = self.sde.sample_t(key1, t_shape)

        # Compute marginals
        mean_coeff = self.sde.marginal_mean_coeff(t)
        std_t = self.sde.marginal_std(t)

        # Noise and construct x_t
        noise = jax.random.normal(key2, x_1.shape)
        x_t = mean_coeff * x_1 + std_t * noise

        return SMPathSample(
            x_1=x_1,
            x_t=x_t,
            t=t,
            noise=noise,
            std_t=std_t,
        )

    def sample_prior(self, key: Array, shape: Any) -> Array:
        """
        Sample from the prior distribution.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the samples to generate, should be (nsamples, ndim).

        Returns
        -------
            Array
                Samples from the prior distribution.
        """
        return self.sde.sample_prior(key, shape)

    def get_loss_fn(self) -> Callable:
        r"""
        Returns the loss function for score matching training.

        The loss is the denoising score matching objective:

        .. math::
            g(t)^2 \left\| s_\theta(x_t, t) - \left(-\frac{\epsilon}{\sigma(t)}\right) \right\|^2

        where :math:`-\epsilon / \sigma(t)` is the true score :math:`\nabla_x \log p(x_t | x_0)`.

        Returns
        -------
            Callable
                Loss function.
        """
        sde = self.sde

        def loss_fn(
            F: Callable,
            batch: tuple,
            condition_mask: Any = None,
            model_extras: dict = {},
        ) -> Array:
            (x_1, x_t, t, noise, std_t) = batch

            # Score target: -noise / std_t = nabla_x log p(x_t | x_0)
            score_target = -noise / std_t

            # Weight for MLE: g(t)^2
            w = sde.weight(t)

            if condition_mask is not None:
                condition_mask = jnp.broadcast_to(condition_mask, x_1.shape)
                x_t = jnp.where(condition_mask, x_1, x_t)

            # Model predicts score
            score_pred = F(obs=x_t, t=t, **model_extras)

            loss = w * (score_pred - score_target) ** 2
            if condition_mask is not None:
                loss = jnp.where(condition_mask, 0.0, loss)

            return jnp.mean(jnp.sum(loss, axis=tuple(range(1, len(x_1.shape)))))

        return loss_fn
