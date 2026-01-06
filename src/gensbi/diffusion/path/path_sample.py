"""
Path sample data structures for diffusion models.

This module defines data structures for representing samples along the diffusion
probability path, including EDM path samples from "Elucidating the Design Space 
of Diffusion-Based Generative Models" (Karras et al., 2022).
"""
from dataclasses import dataclass, field
from jax import Array
from typing import Tuple


@dataclass
class EDMPathSample:
    r"""Represents a sample of a diffusion generated probability path.

    Attributes:
        x_1 (Array): the target sample :math:`X_1`.
        sigma (Array): the noise scale :math:`t`.
        x_t (Array): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
    """

    x_1: Array = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    sigma: Array = field(metadata={"help": "noise scale sigma (batch_size, ...)."})
    x_t: Array = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )

    def get_batch(self) -> Tuple[Array, Array, Array]:
        r"""
        Returns the batch as a tuple (x_1, x_t, sigma).

        Returns
        -------
            Tuple[Array, Array, Array]: The target sample, the noisy sample, and the noise scale.
        """
        return self.x_1, self.x_t, self.sigma
