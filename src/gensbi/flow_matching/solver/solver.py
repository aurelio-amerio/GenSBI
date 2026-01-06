"""
Abstract solver interface for flow matching.

This module defines the abstract base class for solvers used in flow matching algorithms.
"""
from abc import ABC, abstractmethod
from jax import Array


class Solver(ABC):
    """Abstract base class for flow matching solvers."""

    @abstractmethod
    def sample(self, x_0: Array) -> Array:
        """
        Sample from the solver given initial conditions.
        
        Parameters
        ----------
            x_0: Initial conditions for the solver.
            
        Returns
        -------
            Sampled output from the solver.
        """
        ...  # pragma: no cover
