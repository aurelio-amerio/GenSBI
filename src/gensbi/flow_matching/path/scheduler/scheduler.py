"""
Schedulers for flow-matching paths.

It is advised to use the `CondOTScheduler` for optimal performance with conditional flow matching.

"""

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
        Parameters
        ----------
            t : Array
                times in [0,1], shape (...).

        Returns
        -------
            SchedulerOutput
                :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...  # pragma: no cover

    @abstractmethod
    def snr_inverse(self, snr: Array) -> Array:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Parameters
        ----------
            snr : Array
                The signal-to-noise, shape (...)

        Returns
        -------
            Array
                t, shape (...)
        """
        ...  # pragma: no cover


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Array) -> SchedulerOutput:
        """Scheduler for convex paths.

        Parameters
        ----------
            t : Array
                times in [0,1], shape (...).

        Returns
        -------
            SchedulerOutput
                :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...  # pragma: no cover

    @abstractmethod
    def kappa_inverse(self, kappa: Array) -> Array:
        """
        Computes :math:`t` from :math:`\kappa_t`.

        Parameters
        ----------
            kappa : Array
                :math:`\kappa`, shape (...)

        Returns
        -------
            Array
                t, shape (...)
        """
        ...  # pragma: no cover

    def snr_inverse(self, snr: Array) -> Array:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Parameters
        ----------
            snr : Array
                The signal-to-noise, shape (...)

        Returns
        -------
            Array
                t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """
    Conditional Optimal Transport (CondOT) Scheduler.
    
    This scheduler provides a linear interpolation path with alpha_t = t and sigma_t = 1 - t,
    which is optimal for conditional optimal transport flow matching.
    """

    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Compute scheduler outputs for given times.
        
        Parameters
        ----------
            t: Times in [0,1], shape (...).
            
        Returns
        -------
            Scheduler output containing alpha_t, sigma_t, and their derivatives.
        """
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-jnp.ones_like(t),
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        """
        Compute t from kappa.
        
        Parameters
        ----------
            kappa: Kappa values, shape (...).
            
        Returns
        -------
            Time values, shape (...).
        """
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """
    Polynomial Convex Scheduler.
    
    This scheduler uses polynomial interpolation with alpha_t = t^n and sigma_t = 1 - t^n.
    
    Parameters
    ----------
        n: The polynomial degree, must be positive.
    """

    def __init__(self, n: Union[float, int]) -> None:
        """
        Initialize the polynomial convex scheduler.
        
        Parameters
        ----------
            n: Polynomial degree, must be a positive float or int.
            
        Raises
        ------
            AssertionError
                If n is not a float/int or if n is not positive.
        """
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."
        self.n = n

    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Compute scheduler outputs for given times.
        
        Parameters
        ----------
            t: Times in [0,1], shape (...).
            
        Returns
        -------
            Scheduler output containing alpha_t, sigma_t, and their derivatives.
        """
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        """
        Compute t from kappa.
        
        Parameters
        ----------
            kappa: Kappa values, shape (...).
            
        Returns
        -------
            Time values, shape (...).
        """
        return jnp.power(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """
    Variance Preserving (VP) Scheduler.
    
    This scheduler follows the variance-preserving SDE formulation commonly used in
    diffusion models, with configurable beta_min and beta_max parameters.
    
    Parameters
    ----------
        beta_min: Minimum beta value. Defaults to 0.1.
        beta_max: Maximum beta value. Defaults to 20.0.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """
        Initialize the VP scheduler.
        
        Parameters
        ----------
            beta_min: Minimum beta value.
            beta_max: Maximum beta value.
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Compute scheduler outputs for given times.
        
        Parameters
        ----------
            t: Times in [0,1], shape (...).
            
        Returns
        -------
            Scheduler output containing alpha_t, sigma_t, and their derivatives.
        """
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
        """
        Compute t from signal-to-noise ratio.
        
        Parameters
        ----------
            snr: The signal-to-noise ratio, shape (...).
            
        Returns
        -------
            Time values, shape (...).
        """
        T = -jnp.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + jnp.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    """
    Linear Variance Preserving Scheduler.
    
    A linear variance-preserving scheduler where alpha_t = t and sigma_t = sqrt(1 - t^2).
    """

    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Compute scheduler outputs for given times.
        
        Parameters
        ----------
            t: Times in [0,1], shape (...).
            
        Returns
        -------
            Scheduler output containing alpha_t, sigma_t, and their derivatives.
        """
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Array) -> Array:
        """
        Compute t from signal-to-noise ratio.
        
        Parameters
        ----------
            snr: The signal-to-noise ratio, shape (...).
            
        Returns
        -------
            Time values, shape (...).
        """
        return jnp.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """
    Cosine Scheduler.
    
    A cosine-based scheduler where alpha_t = sin(pi/2 * t) and sigma_t = cos(pi/2 * t).
    This provides a smooth interpolation between distributions.
    """

    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Compute scheduler outputs for given times.
        
        Parameters
        ----------
            t: Times in [0,1], shape (...).
            
        Returns
        -------
            Scheduler output containing alpha_t, sigma_t, and their derivatives.
        """
        return SchedulerOutput(
            alpha_t=jnp.sin(jnp.pi / 2 * t),
            sigma_t=jnp.cos(jnp.pi / 2 * t),
            d_alpha_t=jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t),
            d_sigma_t=-jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t),
        )

    def snr_inverse(self, snr: Array) -> Array:
        """
        Compute t from signal-to-noise ratio.
        
        Parameters
        ----------
            snr: The signal-to-noise ratio, shape (...).
            
        Returns
        -------
            Time values, shape (...).
        """
        return 2.0 * jnp.arctan(snr) / jnp.pi
