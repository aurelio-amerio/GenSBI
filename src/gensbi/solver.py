"""
Abstract base class for solvers.
"""

from abc import ABC, abstractmethod
from typing import Any

from jax import Array


class Solver(ABC):
    """Abstract base class for generative model solvers."""

    @abstractmethod
    def sample(self, *args, **kwargs) -> Array:
        """
        Sample from the solver.

        Parameters
        ----------
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        Array
            Sampled output from the solver.
        """
        ...  # pragma: no cover

    def get_unnormalized_logprob(self, *args, **kwargs):
        """Return a callable that computes the unnormalized log-probability.

        Only supported by solvers that can evaluate the continuous
        change-of-variables formula (e.g. ``ODESolver``).

        Raises
        ------
        NotImplementedError
            If the solver does not support log-probability computation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support log-probability computation."
        )

    def unnormalized_logprob(self, *args, **kwargs):
        """Compute the unnormalized log-probability for given samples.

        Only supported by solvers that can evaluate the continuous
        change-of-variables formula (e.g. ``ODESolver``).

        Raises
        ------
        NotImplementedError
            If the solver does not support log-probability computation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support log-probability computation."
        )

