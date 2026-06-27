"""NLE posterior target: build log-densities from a trained likelihood flow + prior.

The flow is NLE-trained (``obs = x``, ``cond = theta``), so ``flow.log_prob(x, theta)``
is ``log q(x | theta)``. ``NLEPosterior`` turns ``(flow, prior, x_o)`` into a
``PosteriorTarget`` (separate log-prior / log-likelihood / log-posterior closures),
which a ``Sampler`` consumes. The flow params are frozen constants inside the
closures; only ``theta`` is traced/differentiated.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from gensbi.utils.math import _expand_dims


@dataclass(frozen=True)
class PosteriorTarget:
    """Log-densities for one observation ``x_o``. All callables take flat ``theta (dim,)``."""
    log_prior: Callable
    log_likelihood: Callable
    log_posterior: Callable
    prior: object
    dim: int


class NLEPosterior:
    """Amortized NLE posterior over a trained likelihood flow.

    Parameters
    ----------
    flow : object
        Exposes ``log_prob(x, cond) -> (B,)`` (an NLE-trained ``MAFlow``/``TarFlow``).
    prior : numpyro.distributions.Distribution
        Prior over theta; ``prior.log_prob(theta)`` is a scalar and
        ``prior.sample(key, ())`` returns ``(dim,)``.
    structured_obs : bool
        If True, ``x_o`` keeps its (image/field) shape instead of being flattened.
    """

    def __init__(self, flow, prior, *, structured_obs: bool = False):
        self.flow = flow
        self.prior = prior
        self.structured_obs = structured_obs

    def build_target(self, x_o) -> PosteriorTarget:
        if self.structured_obs:
            x_o = jnp.asarray(x_o)
        else:
            x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))   # (dim_x,)
        flow = self.flow
        prior = self.prior
        dim = int(prior.event_shape[0])

        def log_prior(theta):
            return prior.log_prob(jnp.asarray(theta))

        def log_likelihood(theta):
            theta = jnp.asarray(theta)
            return flow.log_prob(x_o[None], theta[None, :])[0]

        def log_posterior(theta):
            return log_likelihood(theta) + log_prior(theta)

        return PosteriorTarget(
            log_prior=log_prior, log_likelihood=log_likelihood,
            log_posterior=log_posterior, prior=prior, dim=dim,
        )

    def sample(self, key, x_o, sampler=None, *, return_info=False):
        """Draw posterior samples. Returns (n, dim, 1), or (samples, info) if return_info."""
        from gensbi.inference.samplers import MCLMC
        sampler = sampler if sampler is not None else MCLMC()
        target = self.build_target(x_o)
        samples, info = sampler.run(key, target)
        samples = _expand_dims(samples)          # (n, dim) -> (n, dim, 1)
        return (samples, info) if return_info else samples
