"""
SDE prior distributions for score matching.

Provides prior classes with ``sample`` and ``log_prob`` support,
following the same pattern as ``StandardNormalPrior`` from flow matching.
Ready for use in both sampling (Phase 1) and log_prob (Phase 2).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Any
import warnings

import numpyro.distributions as dist


class VPPrior:
    r"""VP-SDE prior: standard normal :math:`\mathcal{N}(0, I)`.

    At :math:`t = T` the VP marginal converges to :math:`\mathcal{N}(0, I)`,
    identical to the flow matching prior.

    Examples
    --------
    >>> prior = VPPrior()
    >>> key = jax.random.PRNGKey(0)
    >>> x = prior.sample(key, (4, 3, 1))
    >>> assert x.shape == (4, 3, 1)
    >>> lp = prior.log_prob(x)
    >>> assert lp.shape == (4,)
    """

    def sample(self, key: Array, shape: Any) -> Array:
        """Draw samples from :math:`\\mathcal{N}(0, I)`."""
        warnings.warn(
            "VPPrior is deprecated. Use make_gaussian_prior() from gensbi.prior instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return jax.random.normal(key, shape)

    def log_prob(self, x: Array) -> Array:
        r"""Log-probability under :math:`\mathcal{N}(0, I)` for non-batch dims.

        Parameters
        ----------
        x : Array
            Input of shape ``(batch, features, channels)`` or
            ``(features, channels)``.

        Returns
        -------
        Array
            Log-probability per sample, shape ``(batch,)`` or scalar.
        """
        event_shape = x.shape[-2:]  # (features, channels)
        p0 = dist.Independent(
            dist.Normal(
                loc=jnp.zeros(event_shape),
                scale=jnp.ones(event_shape),
            ),
            reinterpreted_batch_ndims=len(event_shape),
        )
        return p0.log_prob(x)


class VEPrior:
    r"""VE-SDE prior: :math:`\mathcal{N}(0, \sigma_{\max}^2 I)`.

    At :math:`t = T` the VE marginal has std :math:`\sigma_{\max}`,
    so the prior is the wider Gaussian.

    Parameters
    ----------
    sigma_max : float
        Maximum noise level from the VE scheduler.

    Examples
    --------
    >>> prior = VEPrior(sigma_max=15.0)
    >>> key = jax.random.PRNGKey(0)
    >>> x = prior.sample(key, (4, 3, 1))
    >>> assert x.shape == (4, 3, 1)
    >>> lp = prior.log_prob(x)
    >>> assert lp.shape == (4,)
    """

    def __init__(self, sigma_max: float):
        warnings.warn(
            "VEPrior is deprecated. Use make_gaussian_prior(sigma=sigma_max) from gensbi.prior instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.sigma_max = sigma_max

    def sample(self, key: Array, shape: Any) -> Array:
        r"""Draw samples from :math:`\mathcal{N}(0, \sigma_{\max}^2 I)`."""
        return self.sigma_max * jax.random.normal(key, shape)

    def log_prob(self, x: Array) -> Array:
        r"""Log-probability under :math:`\mathcal{N}(0, \sigma_{\max}^2 I)`.

        Parameters
        ----------
        x : Array
            Input of shape ``(batch, features, channels)`` or
            ``(features, channels)``.

        Returns
        -------
        Array
            Log-probability per sample, shape ``(batch,)`` or scalar.
        """
        event_shape = x.shape[-2:]  # (features, channels)
        p0 = dist.Independent(
            dist.Normal(
                loc=jnp.zeros(event_shape),
                scale=jnp.full(event_shape, self.sigma_max),
            ),
            reinterpreted_batch_ndims=len(event_shape),
        )
        return p0.log_prob(x)
