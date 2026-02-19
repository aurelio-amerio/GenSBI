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
        ...
