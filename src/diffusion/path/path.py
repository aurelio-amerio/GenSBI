from abc import ABC

class ProbPath(ABC):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        return

    def sample_prior(self, key, shape):
        """
        Sample from the prior distribution.

        Args:
            key: JAX random key.
            shape: Shape of the samples to generate, should be (nsamples, ndim).

        Returns:
            Array: Samples from the prior distribution, shape (nsamples, ndim).
        """
        return self.scheduler.sample_prior(key, shape)

    @property 
    def name(self):
        return self.scheduler.name