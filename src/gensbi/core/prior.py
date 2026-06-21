"""Gaussian prior factory for GenSBI pipelines."""

import jax.numpy as jnp
import numpyro.distributions as dist


def make_gaussian_prior(dim, ch=None, *, mu=0.0, sigma=1.0):
    """Create a Gaussian prior as a numpyro distribution.

    Two call forms:

    - ``make_gaussian_prior(dim, ch)`` — legacy rank-2 form,
      ``event_shape=(dim, ch)``.
    - ``make_gaussian_prior(event_shape)`` — a single tuple of any rank,
      e.g. ``(H, W, C)`` for pixel-space fields.

    ``mu`` and ``sigma`` are keyword-only: ``make_gaussian_prior(H, W, C)``
    raises ``TypeError`` instead of silently reading ``C`` as the mean.

    Returns
    -------
    dist.Independent
        ``Independent(Normal(loc, scale), len(event_shape))``.
    """
    if ch is None:
        if not isinstance(dim, (tuple, list)):
            raise TypeError(
                "pass (dim, ch) as two ints or a single event_shape tuple, "
                f"got dim={dim!r} with ch=None"
            )
        event_shape = tuple(dim)
    else:
        event_shape = (dim, ch)
    loc = jnp.full(event_shape, mu)
    scale = jnp.full(event_shape, sigma)
    return dist.Independent(dist.Normal(loc, scale), len(event_shape))


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
