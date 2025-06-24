from abc import ABC, abstractmethod
import jax
from jax import Array
from jax import numpy as jnp


from gensbi.diffusion.path.path import ProbPath
from gensbi.diffusion.path.path_sample import PathSample


class EDMPath(ProbPath):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        assert self.scheduler.name in [
            "EDM",
            "EDM-VP",
            "EDM-VE",
        ], f"Scheduler must be one of ['EDM', 'EDM-VP', 'EDM-VE'], got {self.scheduler.name}."
        return

    def sample(self, key, x_1, sigma) -> PathSample:
        noise = self.scheduler.sample_noise(key, x_1.shape, sigma)
        x_t = x_1 + noise
        return PathSample(
            x_1=x_1,
            sigma=sigma,
            x_t=x_t,
        )

    def sample_sigma(self, key, batch_size) -> Array:
        """
        Sample the noise scale sigma from the scheduler.

        Args:
            key: JAX random key.
            shape: Shape of the samples to generate.

        Returns:
            Array: Samples of sigma, shape (batch_size, ...).
        """
        return self.scheduler.sample_sigma(key, batch_size)[..., None]

    def get_loss_fn(self):
        """
        Returns the loss function for the EDM path.

        # Args:
        #     F: The model, in the form F(x, t, *args, **kwargs).
        #     x0: The original image or data point.
        #     loss_mask: An optional mask to apply to the loss for conditioning.
        #     *args: Additional positional arguments for the model F.
        #     key: JAX random key for stochastic operations.
        #     **kwargs: Additional keyword arguments for the model F.
        """
        return self.scheduler.get_loss_fn()
