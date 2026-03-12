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

from gensbi.diffusion.path.path import ProbPath
from gensbi.diffusion.path.path_sample import SMPathSample


class SMPath(ProbPath):
    """
    Score Matching probability path.

    This class constructs noised samples for standard score matching training
    using the forward SDE's marginal distributions.

    The noising is: x_t = mean_coeff(t) * x_1 + std(t) * epsilon

    Parameters
    ----------
        scheduler: The SDE scheduler object (VPSmScheduler or VESmScheduler).

    Example:
        .. code-block:: python

            from gensbi.diffusion.path.sm_path import SMPath
            from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler
            import jax, jax.numpy as jnp
            scheduler = VPSmScheduler()
            path = SMPath(scheduler)
            key = jax.random.PRNGKey(0)
            x_1 = jax.random.normal(key, (32, 2))
            x_0 = jax.random.normal(jax.random.PRNGKey(1), (32, 2))
            t = jnp.ones((32, 1, 1)) * 0.5
            sample = path.sample(x_0, x_1, t)
    """

    def __init__(self, scheduler) -> None:
        """
        Initialize the SMPath with an SDE scheduler.

        Parameters
        ----------
            scheduler: The SDE scheduler object.

        Raises
        ------
            AssertionError
                If scheduler name is not one of 'SM-VP' or 'SM-VE'.
        """
        self.scheduler = scheduler
        assert self.scheduler.name in [
            "SM-VP",
            "SM-VE",
        ], f"SDE must be one of ['SM-VP', 'SM-VE'], got {self.scheduler.name}."
        return

    def sample(self, x_0: Array, x_1: Array, t: Array) -> SMPathSample:
        r"""
        Sample from the score matching probability path.

        Constructs ``x_t = mean_coeff(t) * x_1 + std(t) * x_0`` where
        ``x_0`` is standard normal noise.

        Parameters
        ----------
            x_0 : Array
                Source noise sample from N(0, 1), shape (batch_size, ...).
            x_1 : Array
                Target data point, shape (batch_size, ...).
            t : Array
                Diffusion time, shape (batch_size, 1, ...).
                Use :meth:`sample_t` to sample appropriate times.

        Returns
        -------
            SMPathSample
                A sample from the SM path.
        """
        # Compute marginals
        mean_coeff = self.scheduler.marginal_mean_coeff(t)
        std_t = self.scheduler.marginal_std(t)

        # Construct x_t from pre-sampled noise x_0
        x_t = mean_coeff * x_1 + std_t * x_0

        return SMPathSample(
            x_1=x_1,
            x_t=x_t,
            t=t,
            noise=x_0,
            std_t=std_t,
        )

    def sample_t(self, key: Array, shape) -> Array:
        """
        Sample diffusion times from the SDE scheduler.

        Analogous to :meth:`EDMPath.sample_sigma`.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : tuple
                Shape of the time samples to generate.

        Returns
        -------
            Array
                Sampled diffusion times.
        """
        return self.scheduler.sample_t(key, shape)

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
        scheduler = self.scheduler

        def loss_fn(
            F: Callable,
            batch: tuple,
            condition_mask: Any = None,
            weights: Any = None,
            model_extras: dict = None,
        ) -> Array:
            if model_extras is None:
                model_extras = {}
            (x_1, x_t, t, noise, std_t) = batch

            # Score target: -noise / std_t = nabla_x log p(x_t | x_0)
            score_target = -noise / std_t

            # Weight for MLE: g(t)^2
            w = scheduler.weight(t)

            if condition_mask is not None:
                condition_mask = jnp.broadcast_to(condition_mask, x_1.shape)
                x_t = jnp.where(condition_mask, x_1, x_t)

            # Model predicts score
            score_pred = F(obs=x_t, t=t, **model_extras)

            if weights is not None:
                weights = jnp.broadcast_to(weights, x_1.shape)
            else:
                weights = jnp.ones_like(x_1)

            loss = weights * w * (score_pred - score_target) ** 2
            if condition_mask is not None:
                loss = jnp.where(condition_mask, 0.0, loss)

            return jnp.mean(jnp.sum(loss, axis=tuple(range(1, len(x_1.shape)))))

        return loss_fn
