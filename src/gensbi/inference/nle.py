"""NLE posterior: a trained likelihood flow + a prior -> NumPyro NUTS.

The flow is NLE-trained (``obs = x``, ``cond = theta``), so ``flow.log_prob(x, theta)``
is ``log q(x | theta)``. We form the (unnormalized) posterior potential
``U(theta) = -(log q(x_o | theta) + log p(theta))`` and run NUTS on it directly
(potential-function route; NOT a numpyro Distribution wrapper). The flow's params
are frozen constants inside the potential; only ``theta`` is traced/differentiated.
"""

import jax
import jax.numpy as jnp

from gensbi.utils.math import _expand_dims


class NLEPosterior:
    """Amortized NLE posterior over a trained likelihood flow.

    Parameters
    ----------
    flow : object
        Anything exposing ``log_prob(x, cond) -> (B,)`` with x the observation
        and cond the parameter (an NLE-trained flow model (``MAFlow``/``TarFlow``)).
    prior : numpyro.distributions.Distribution
        Prior over theta; ``prior.log_prob(theta)`` returns a scalar and
        ``prior.sample(key, ())`` returns ``(dim_theta,)``.
    num_warmup, num_samples, num_chains : int
        NUTS defaults (overridable per ``sample`` call via ``nsamples``).
    """

    def __init__(self, flow, prior, *, num_warmup=500, num_samples=1000,
                 num_chains=1, structured_obs=False):
        self.flow = flow
        self.prior = prior
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.structured_obs = structured_obs

    def potential(self, x_o):
        """Return ``U(theta) = -(log q(x_o|theta) + log p(theta))`` for one x_o."""
        if self.structured_obs:
            x_o = jnp.asarray(x_o)
        else:
            x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))   # (dim_x,)
        flow = self.flow
        prior = self.prior

        def U(theta):
            theta = jnp.asarray(theta)
            log_like = flow.log_prob(x_o[None], theta[None, :])[0]
            log_prior = prior.log_prob(theta)
            return -(log_like + log_prior)

        return U

    def sample(self, key, x_o, nsamples=None):
        """Draw posterior samples via NUTS. Returns ``(n, dim_theta, 1)``.

        Uses the potential-function route: ``init_params`` is a single draw from
        the prior, so ``mcmc.get_samples()`` returns a raw ``(n, dim_theta)`` array.
        """
        from numpyro.infer import MCMC, NUTS

        n = self.num_samples if nsamples is None else nsamples
        potential = self.potential(x_o)
        kernel = NUTS(potential_fn=potential)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=n,
                    num_chains=self.num_chains, progress_bar=False)
        key, key_init = jax.random.split(key)
        init_params = self.prior.sample(key_init, ())        # (dim_theta,)
        mcmc.run(key, init_params=init_params)
        samples = jnp.asarray(mcmc.get_samples())            # (n, dim_theta)
        return _expand_dims(samples)                         # (n, dim_theta, 1)
