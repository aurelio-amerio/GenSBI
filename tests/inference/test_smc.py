# tests/inference/test_smc.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference import NLEPosterior, TemperedSMC


class GaussianMock:
    def log_prob(self, x, cond):
        diff = (x - cond).reshape(x.shape[0], -1)     # flatten all non-batch dims
        return -0.5 * jnp.sum(diff ** 2, axis=-1)     # (B,)


class BimodalMock:
    """log q(x | theta) = mixture of N(theta; +mu, 0.5 I) and N(theta; -mu, 0.5 I).

    Independent of the observation x; posterior under a broad prior is bimodal at +/-mu.
    """
    def __init__(self, mu=3.0, sigma=0.5):
        self.mu, self.sigma = mu, sigma

    def log_prob(self, x, cond):
        cf = cond.reshape(cond.shape[0], -1)           # flatten all non-batch dims
        a = -0.5 * jnp.sum(((cf - self.mu) / self.sigma) ** 2, axis=-1)
        b = -0.5 * jnp.sum(((cf + self.mu) / self.sigma) ** 2, axis=-1)
        return jax.scipy.special.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)


def test_smc_hmc_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(0), x_o,
                    sampler=TemperedSMC(inner_kernel="hmc", num_particles=2000,
                                        inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    assert s.shape == (2000, dim)
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.2)


def test_smc_hmc_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    s = post.sample(jax.random.PRNGKey(1), jnp.zeros(dim),
                    sampler=TemperedSMC(inner_kernel="hmc", num_particles=2000,
                                        inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    # both modes populated (a single MCMC chain would capture only one)
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_smc_info_has_log_evidence():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    _, info = post.sample(jax.random.PRNGKey(2), jnp.array([1.0, -1.0]),
                          sampler=TemperedSMC(inner_kernel="hmc", num_particles=1000,
                                              inner_step_size=0.5,
                                              inner_num_integration_steps=10),
                          return_info=True)
    assert jnp.isfinite(info.log_evidence)
    assert info.num_temperature_steps > 0
    assert jnp.isclose(info.final_tempering_param, 1.0, atol=1e-6)


def test_smc_mclmc_is_the_default_inner_kernel():
    assert TemperedSMC().inner_kernel == "mclmc"


def test_smc_mclmc_recovers_both_modes():
    dim = 2
    post = NLEPosterior(BimodalMock(mu=3.0), make_gaussian_prior((dim,), sigma=5.0))
    # default inner kernel == adjusted MCLMC
    s = post.sample(jax.random.PRNGKey(7), jnp.zeros(dim),
                    sampler=TemperedSMC(num_particles=2000, inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    frac_pos = jnp.mean(jnp.all(s > 0, axis=1).astype(float))
    frac_neg = jnp.mean(jnp.all(s < 0, axis=1).astype(float))
    assert frac_pos > 0.3 and frac_neg > 0.3


def test_smc_mclmc_analytic_gaussian_recovery():
    dim = 2
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((dim,)))
    x_o = jnp.array([1.0, -1.0])
    s = post.sample(jax.random.PRNGKey(8), x_o,
                    sampler=TemperedSMC(num_particles=2000, inner_step_size=0.5,
                                        inner_num_integration_steps=10))[..., 0]
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.2)


def test_smc_unknown_kernel_raises():
    post = NLEPosterior(GaussianMock(), make_gaussian_prior((2,)))
    with pytest.raises(ValueError):
        post.sample(jax.random.PRNGKey(4), jnp.array([1.0, -1.0]),
                    sampler=TemperedSMC(inner_kernel="foo", num_particles=100))
