from abc import ABC, abstractmethod
import jax
from jax import Array
from jax import numpy as jnp
from typing import Callable
import chex

from gensbi.diffusion.path.path import ProbPath
from gensbi.diffusion.path.path_sample import EDMPathSample


class EDMPath(ProbPath):
    def __init__(self, scheduler) -> None:
        r"""
        Initialize the EDMPath with a scheduler.

        Args:
            scheduler: The scheduler object.

        Example:
            .. code-block:: python

                from gensbi.diffusion.path import EDMPath
                from gensbi.diffusion.path.scheduler import EDMScheduler
                import jax, jax.numpy as jnp
                scheduler = EDMScheduler()
                path = EDMPath(scheduler)
                key = jax.random.PRNGKey(0)
                x_1 = jax.random.normal(key, (32, 2))
                sigma = jnp.ones((32, 1))
                sample = path.sample(key, x_1, sigma)
                print(sample.x_t.shape)
                # (32, 2)

        """
        self.scheduler = scheduler
        assert self.scheduler.name in [
            "EDM",
            "EDM-VP",
            "EDM-VE",
        ], f"Scheduler must be one of ['EDM', 'EDM-VP', 'EDM-VE'], got {self.scheduler.name}."
        return

    def sample(self, key: Array, x_1: Array, sigma: Array) -> EDMPathSample:
        r"""
        Sample from the EDM probability path.

        Args:
            key (Array): JAX random key.
            x_1 (Array): Target data point, shape (batch_size, ...).
            sigma (Array): Noise scale, shape (batch_size, ...).

        Returns:
            PathSample: A sample from the EDM path.
        """
        noise = self.scheduler.sample_noise(key, x_1.shape, sigma)
        x_t = x_1 + noise
        return EDMPathSample(
            x_1=x_1,
            sigma=sigma,
            x_t=x_t,
        )

    def sample_sigma(self, key: Array, batch_size: int) -> Array:
        r"""
        Sample the noise scale sigma from the scheduler.

        Args:
            key (Array): JAX random key.
            batch_size (int): Number of samples to generate.

        Returns:
            Array: Samples of sigma, shape (batch_size, ...).
        """
        return self.scheduler.sample_sigma(key, batch_size)[..., None]

    def get_loss_fn(self) -> Callable:
        r"""
        Returns the loss function for the EDM path.

        Returns:
            Callable: The loss function as provided by the scheduler.
        """
        return self.scheduler.get_loss_fn()
