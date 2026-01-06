"""
Abstract solver interface for diffusion models.

This module defines the abstract base class for solvers used in diffusion model sampling.
"""
from abc import ABC, abstractmethod

from jax import Array


class Solver(ABC):
    """Abstract base class for diffusion model solvers."""

    @abstractmethod
    def sample(self, key, x_1: Array) -> Array:
        """
        Sample from the diffusion solver given target conditions.
        
        Parameters
        ----------
            key: JAX random key for stochastic operations.
            x_1: Target conditions for the solver.
            
        Returns
        -------
            Sampled output from the solver.
        """
        ...  # pragma: no cover
