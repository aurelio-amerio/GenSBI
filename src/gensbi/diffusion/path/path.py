from abc import ABC, abstractmethod
from jax import Array
from gensbi.diffusion.path.path_sample import EDMPathSample
from typing import Any

class ProbPath(ABC):
    def __init__(self, scheduler: Any) -> None:
        """
        Initialize the probability path.

        Args:
            scheduler: Scheduler object.
        """
        self.scheduler = scheduler
        return

    def sample_prior(self, key: Array, shape: Any) -> Array:
        """
        Sample from the prior distribution.

        Args:
            key (Array): JAX random key.
            shape (Any): Shape of the samples to generate, should be (nsamples, ndim).

        Returns:
            Array: Samples from the prior distribution, shape (nsamples, ndim).
        """
        return self.scheduler.sample_prior(key, shape)

    @property 
    def name(self) -> str:
        """
        Returns the name of the scheduler.

        Returns:
            str: Scheduler name.
        """
        return self.scheduler.name

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> "EDMPathSample":
        """
        Abstract method to sample from the probability path.

        Returns:
            PathSample: Sample from the path.
        """
        ...