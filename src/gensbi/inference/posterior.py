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
    """Log-densities for one observation ``x_o``.

    Frozen dataclass produced by :meth:`NLEPosterior.build_target`.  All
    callables accept a flat parameter vector ``theta`` of shape ``(dim,)``.

    Parameters
    ----------
    log_prior : Callable
        Log-prior density.  Signature: ``log_prior(theta) -> float``.
    log_likelihood : Callable
        Log-likelihood ``log q(x_o | theta)`` from the NLE-trained flow.
        Signature: ``log_likelihood(theta) -> float``.
    log_posterior : Callable
        Unnormalised log-posterior ``log_likelihood(theta) + log_prior(theta)``.
        Signature: ``log_posterior(theta) -> float``.
    prior : object
        Prior distribution; must expose ``sample(key, shape)`` and
        ``log_prob(theta)``.
    dim : int
        Dimensionality of the parameter space.
    """

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
    structured_obs : bool, optional
        If ``True``, ``x_o`` keeps its (image/field) shape instead of being
        flattened.  Default is ``False``.
    """

    def __init__(self, flow, prior, *, structured_obs: bool = False):
        self.flow = flow
        self.prior = prior
        self.structured_obs = structured_obs

    def build_target(self, x_o) -> PosteriorTarget:
        """Build a posterior target for a single observation.

        Parameters
        ----------
        x_o : Array
            Observed data.  Squeezed to shape ``(dim_x,)`` unless
            ``structured_obs=True``.

        Returns
        -------
        PosteriorTarget
            Frozen log-density container for ``x_o``.
        """
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
        """Draw posterior samples for a single observation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_o : Array
            Observed data passed to :meth:`build_target`.
        sampler : Sampler or None, optional
            Sampler instance to use.  If ``None``, defaults to
            :class:`~gensbi.inference.samplers.MCLMC`.  Default is ``None``.
        return_info : bool, optional
            If ``True``, return a ``(samples, info)`` tuple instead of just
            ``samples``.  Default is ``False``.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(n, dim, 1)``.  When
            ``return_info=False`` (the default), this is the only return value.
        info : object
            Sampler-specific info object
            (:class:`~gensbi.inference.samplers.MclmcInfo` or
            :class:`~gensbi.inference.samplers.SmcInfo`).  Only present when
            ``return_info=True``.
        """
        from gensbi.inference.samplers import MCLMC
        sampler = sampler if sampler is not None else MCLMC()
        target = self.build_target(x_o)
        samples, info = sampler.run(key, target)
        samples = _expand_dims(samples)          # (n, dim) -> (n, dim, 1)
        return (samples, info) if return_info else samples
