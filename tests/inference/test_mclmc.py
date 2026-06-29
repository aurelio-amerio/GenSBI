import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.models import MAFlow, MAFlowParams
from gensbi.inference import NLEPosterior, MCLMC
from gensbi.inference.samplers import _check_rescale_domain


def test_check_rescale_domain_guard():
    # mu > 0.5 is in-domain (no raise); mu <= 0.5 raises with a clear message.
    _check_rescale_domain(2.0)
    _check_rescale_domain(0.5 + 1e-3)
    for bad in (0.5, 0.25, 0.0):
        with pytest.raises(ValueError, match="L/step_size"):
            _check_rescale_domain(bad)


class GaussianMock:
    """log q(x | theta) = N(x; theta, I); with prior N(0, I) => posterior N(x_o/2, 0.5 I)."""
    def log_prob(self, x, cond):
        diff = (x - cond).reshape(x.shape[0], -1)     # flatten all non-batch dims
        return -0.5 * jnp.sum(diff ** 2, axis=-1)     # (B,)


def test_unadjusted_prior_recovery_real_flow():
    # zero_init flow => q(x|theta) is theta-independent => posterior == prior N(0, I).
    # Exercises MCLMC end-to-end against a real MAFlow.
    dim = 2
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim, cond_dim=dim,
                               n_layers=3, nn_width=16, zero_init=True))
    post = NLEPosterior(flow, make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(9), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=800, num_tuning_steps=600))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), 0.0, atol=0.25)


def test_unadjusted_shape_and_finite():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(0), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=500, num_tuning_steps=500))
    assert s.shape == (500, dim, 1)
    assert jnp.all(jnp.isfinite(s))


def test_unadjusted_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(1), x_o,
                    sampler=MCLMC(adjusted=False, num_samples=3000, num_tuning_steps=2000))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.15)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.2)


def test_unadjusted_multichain_shape():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                    sampler=MCLMC(adjusted=False, num_samples=200, num_tuning_steps=400, num_chains=3))
    assert s.shape == (600, dim, 1)


def test_return_info():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    s, info = post.sample(jax.random.PRNGKey(3), jnp.array([1.0, -1.0]),
                          sampler=MCLMC(adjusted=False, num_samples=200, num_tuning_steps=400),
                          return_info=True)
    assert s.shape == (200, dim, 1)
    assert info.num_samples == 200 and jnp.isfinite(info.L) and jnp.isfinite(info.step_size)


def test_adjusted_is_the_default():
    assert MCLMC().adjusted is True


def test_adjusted_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    # default sampler == adjusted MCLMC; exercised via the one-liner
    s = post.sample(jax.random.PRNGKey(5), x_o,
                    sampler=MCLMC(num_samples=3000, num_tuning_steps=2000))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.15)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.2)


def test_adjusted_reports_acceptance_rate():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    _, info = post.sample(jax.random.PRNGKey(6), jnp.array([1.0, -1.0]),
                          sampler=MCLMC(num_samples=400, num_tuning_steps=600),
                          return_info=True)
    assert 0.0 <= info.acceptance_rate <= 1.0
