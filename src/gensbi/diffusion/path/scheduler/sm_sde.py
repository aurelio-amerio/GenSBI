"""
Standard score matching SDE schedulers.

This module implements variance-preserving (VP) and variance-exploding (VE) SDE
schedulers for standard score matching. These define the forward SDE process and
its marginal distributions, used for training and sampling.

Based on "Score-Based Generative Modeling through Stochastic Differential Equations"
by Song et al., 2021. https://arxiv.org/abs/2011.13456
"""

import abc
import jax
import jax.numpy as jnp
from jax import Array
from typing import Any, Callable


class BaseSMSDE(abc.ABC):
    """
    Base class for standard score matching SDEs.

    Defines the interface for forward SDE processes used in standard
    score matching training and reverse SDE sampling.
    """

    def __init__(self) -> None:
        return

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the SDE."""
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the forward SDE."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def drift(self, x: Array, t: Array) -> Array:
        """
        Drift function f(x, t) of the forward SDE: dx = f(x,t)dt + g(t)dW.

        Parameters
        ----------
            x : Array
                Input data, shape (batch_size, ...).
            t : Array
                Time, shape (batch_size, ...).

        Returns
        -------
            Array
                Drift term, same shape as x.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def diffusion(self, t: Array) -> Array:
        """
        Diffusion coefficient g(t) of the forward SDE: dx = f(x,t)dt + g(t)dW.

        Parameters
        ----------
            t : Array
                Time, shape (batch_size, ...).

        Returns
        -------
            Array
                Diffusion coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def marginal_mean_coeff(self, t: Array) -> Array:
        r"""
        Mean coefficient of the marginal distribution: :math:`\mu(t)` such that
        :math:`\mathbb{E}[x_t | x_0] = \mu(t) x_0`.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Mean coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def marginal_std(self, t: Array) -> Array:
        r"""
        Standard deviation of the marginal distribution at time t.
        :math:`\text{Std}[x_t | x_0] = \sigma(t)`.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Standard deviation.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def sample_t(self, key: Array, shape: Any) -> Array:
        """
        Sample diffusion time for training.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.

        Returns
        -------
            Array
                Sampled times.
        """
        ...  # pragma: no cover

    def sample_prior(self, key: Array, shape: Any) -> Array:
        """
        Sample from the prior distribution (standard normal).

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.

        Returns
        -------
            Array
                Samples from the prior.
        """
        return jax.random.normal(key, shape)

    def weight(self, t: Array) -> Array:
        """
        Loss weight for MLE training. Default is g(t)^2.

        See https://arxiv.org/abs/2101.09258 for justification.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Loss weight.
        """
        return self.diffusion(t) ** 2


class VPSmScheduler(BaseSMSDE):
    r"""
    Variance Preserving (VP) SDE for standard score matching.

    Forward SDE: :math:`dx = -\frac{1}{2} \beta(t) x \, dt + \sqrt{\beta(t)} \, dW`

    Marginal distribution: :math:`x_t = e^{-\frac{1}{2} \alpha(t)} x_0 + \sqrt{1 - e^{-\alpha(t)}} \epsilon`

    where :math:`\alpha(t) = \int_0^t \beta(s) ds = \beta_{\min} t + \frac{1}{2}(\beta_{\max} - \beta_{\min}) t^2`.

    Parameters
    ----------
        beta_min : float
            Minimum value of the beta schedule.
        beta_max : float
            Maximum value of the beta schedule.
        diff_steps : int
            Number of diffusion steps (determines minimum time for training).
    """

    def __init__(
        self,
        beta_min: float = 0.001,
        beta_max: float = 3.0,
        e_s: float = 1e-3,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        self.e_s = e_s
        return

    @property
    def name(self) -> str:
        return "SM-VP"

    @property
    def T(self) -> float:
        return 1.0

    def beta_t(self, t: Array) -> Array:
        """Linear beta schedule."""
        return self.beta_min + self.beta_d * t

    def alpha_t(self, t: Array) -> Array:
        r"""Integral of beta: :math:`\alpha(t) = \beta_{\min} t + \frac{1}{2}(\beta_{\max} - \beta_{\min}) t^2`."""
        return t * self.beta_min + 0.5 * t**2 * self.beta_d

    def drift(self, x: Array, t: Array) -> Array:
        return -0.5 * self.beta_t(t) * x

    def diffusion(self, t: Array) -> Array:
        return jnp.sqrt(self.beta_t(t))

    def marginal_mean_coeff(self, t: Array) -> Array:
        return jnp.exp(-0.5 * self.alpha_t(t))

    def marginal_std(self, t: Array) -> Array:
        return jnp.sqrt(1.0 - jnp.exp(-self.alpha_t(t)))

    def sample_t(self, key: Array, shape: Any) -> Array:
        return jax.random.uniform(key, shape, minval=self.e_s, maxval=1.0)


class VESmScheduler(BaseSMSDE):
    r"""
    Variance Exploding (VE) SDE for standard score matching.

    Forward SDE: :math:`dx = \sigma(t) \sqrt{2 \ln(\sigma_{\max}/\sigma_{\min})} \, dW`

    where :math:`\sigma(t) = \sigma_{\min} (\sigma_{\max}/\sigma_{\min})^t`.

    Marginal distribution: :math:`x_t = x_0 + \sigma(t) \epsilon`.

    Parameters
    ----------
        sigma_min : float
            Minimum noise level.
        sigma_max : float
            Maximum noise level.
        e_s : float
            Minimum time for training (replaces diff_steps).
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 15.0,
        e_s: float = 0.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.e_s = e_s
        self._log_sigma_min = jnp.log(sigma_min)
        self._log_sigma_max = jnp.log(sigma_max)
        return

    @property
    def name(self) -> str:
        return "SM-VE"

    @property
    def T(self) -> float:
        return 1.0

    def sigma(self, t: Array) -> Array:
        r"""Noise level: :math:`\sigma(t) = \sigma_{\min} (\sigma_{\max}/\sigma_{\min})^t`."""
        return jnp.exp(
            self._log_sigma_min + t * (self._log_sigma_max - self._log_sigma_min)
        )

    def drift(self, x: Array, t: Array) -> Array:
        return jnp.zeros_like(x)

    def diffusion(self, t: Array) -> Array:
        return self.sigma(t) * jnp.sqrt(2 * (self._log_sigma_max - self._log_sigma_min))

    def marginal_mean_coeff(self, t: Array) -> Array:
        return jnp.ones_like(t)

    def marginal_std(self, t: Array) -> Array:
        return self.sigma(t)

    def sample_t(self, key: Array, shape: Any) -> Array:
        return jax.random.uniform(key, shape, minval=self.e_s, maxval=1.0)

    def sample_prior(self, key: Array, shape: Any) -> Array:
        r"""
        Sample from the VE prior distribution :math:`\mathcal{N}(0, \sigma_{\max}^2 I)`.

        For the VE SDE, the marginal at :math:`t=T` has std :math:`\sigma_{\max}`,
        so the prior is :math:`\mathcal{N}(0, \sigma_{\max}^2 I)`.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.

        Returns
        -------
            Array
                Samples from the VE prior.
        """
        return self.sigma_max * jax.random.normal(key, shape)
