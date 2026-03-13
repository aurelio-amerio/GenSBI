"""Gaussian prior factory for GenSBI pipelines."""

import jax.numpy as jnp
import numpyro.distributions as dist


def make_gaussian_prior(dim, ch, mu=0.0, sigma=1.0):
    """Create a Gaussian prior as a numpyro distribution.

    Parameters
    ----------
    dim : int
        Feature dimension.
    ch : int
        Channel dimension.
    mu : float
        Prior mean (broadcast to all dimensions).
    sigma : float
        Prior std (broadcast to all dimensions).

    Returns
    -------
    dist.Independent
        ``Independent(Normal(loc, scale), 2)`` with
        ``event_shape=(dim, ch)``.
    """
    loc = jnp.full((dim, ch), mu)
    scale = jnp.full((dim, ch), sigma)
    return dist.Independent(dist.Normal(loc, scale), 2)


def is_gaussian_prior(prior):
    """Check whether a prior is a Gaussian (Normal-based Independent).

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        The prior to check.

    Returns
    -------
    bool
        True if the prior is ``Independent(Normal(...), ...)``.
    """
    if not isinstance(prior, dist.Independent):
        return False
    return isinstance(prior.base_dist, dist.Normal)
