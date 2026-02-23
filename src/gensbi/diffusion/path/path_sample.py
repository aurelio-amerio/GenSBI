"""
Path sample data structures for diffusion models.

This module defines data structures for representing samples along the diffusion
probability path, including EDM path samples from "Elucidating the Design Space
of Diffusion-Based Generative Models" (Karras et al., 2022) and standard score
matching path samples.
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


@dataclass
class SMPathSample:
    r"""Represents a sample from a standard score matching probability path.

    The noising process is: :math:`x_t = \mu(t) x_1 + \sigma(t) \epsilon`

    Attributes:
        x_1 (Array): the clean target sample, shape (batch_size, ...).
        x_t (Array): the noised sample, shape (batch_size, ...).
        t (Array): the diffusion time, shape (batch_size, ...).
        noise (Array): the Gaussian noise epsilon used for noising, shape (batch_size, ...).
        std_t (Array): the marginal standard deviation at time t, shape (batch_size, ...).
    """

    x_1: Array = field(metadata={"help": "clean target samples X_1 (batch_size, ...)."})
    x_t: Array = field(
        metadata={"help": "noised samples x_t, shape (batch_size, ...)."}
    )
    t: Array = field(metadata={"help": "diffusion time t (batch_size, ...)."})
    noise: Array = field(metadata={"help": "Gaussian noise epsilon (batch_size, ...)."})
    std_t: Array = field(metadata={"help": "marginal std at time t (batch_size, ...)."})

    def get_batch(self) -> Tuple[Array, Array, Array, Array, Array]:
        r"""
        Returns the batch as a tuple (x_1, x_t, t, noise, std_t).

        Returns
        -------
            Tuple[Array, Array, Array, Array, Array]:
                The clean sample, noised sample, time, noise, and marginal std.
        """
        return self.x_1, self.x_t, self.t, self.noise, self.std_t
