#FIXME: done

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Array): :math:`\alpha_t`, shape (...).
        sigma_t (Array): :math:`\sigma_t`, shape (...).
        d_alpha_t (Array): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Array): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).
    """
    alpha_t: Array = field(metadata={"help": "alpha_t"})
    sigma_t: Array = field(metadata={"help": "sigma_t"})
    d_alpha_t: Array = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Array = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Array) -> SchedulerOutput:
        r"""
        Args:
            t (Array): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Array) -> Array:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Array): The signal-to-noise, shape (...)

        Returns:
            Array: t, shape (...)
        """
        ...



class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Array) -> SchedulerOutput:
        """Scheduler for convex paths.

        Args:
            t (Array): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Array) -> Array:
        """
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Array): :math:`\kappa`, shape (...)

        Returns:
            Array: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Array) -> Array:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Array): The signal-to-noise, shape (...)

        Returns:
            Array: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t, 
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-jnp.ones_like(t),
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        return kappa



class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(n, (float, int)), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."
        self.n = n

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        return jnp.power(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Array) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=jnp.exp(-0.5 * T),
            sigma_t=jnp.sqrt(1 - jnp.exp(-T)),
            d_alpha_t=-0.5 * dT * jnp.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * jnp.exp(-T) / jnp.sqrt(1 - jnp.exp(-T)),
        )

    def snr_inverse(self, snr: Array) -> Array:
        T = -jnp.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + jnp.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t



class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Array) -> Array:
        return jnp.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=jnp.sin(jnp.pi / 2 * t),
            sigma_t=jnp.cos(jnp.pi / 2 * t),
            d_alpha_t=jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t),
            d_sigma_t=-jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t),
        )

    def snr_inverse(self, snr: Array) -> Array:
        return 2.0 * jnp.arctan(snr) / jnp.pi