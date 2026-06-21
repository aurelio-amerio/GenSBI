import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows import make_maf
from gensbi.inference import NLEPosterior


class GaussianMock:
    """log q(x | theta) = sum_i N(x_i; theta_i, 1) (batched over rows)."""
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)   # (B,)


def test_potential_value_and_grad_real_flow():
    dim = 2
    # zero_init=False so the flow actually depends on theta -> non-trivial grad.
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=dim,
                    n_layers=3, nn_width=16, zero_init=False)
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(flow, prior)

    U = post.potential(jnp.array([0.5, -0.5]))
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == ()
    assert jnp.isfinite(val)
    assert grad.shape == (dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_potential_equals_neg_loglike_plus_logprior():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    x_o = jnp.array([1.0, -1.0])
    theta = jnp.array([0.3, 0.4])
    expected = -(GaussianMock().log_prob(x_o[None], theta[None])[0]
                 + prior.log_prob(theta))
    assert jnp.allclose(post.potential(x_o)(theta), expected)


def test_sample_shape_and_prior_recovery():
    # zero_init=True (default): q(x|theta) is theta-independent (identity flow),
    # so the posterior collapses to the prior. Exercises NUTS + the real flow.
    dim = 2
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=dim,
                    n_layers=3, nn_width=16, zero_init=True)
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(flow, prior, num_warmup=300, num_samples=800)

    s = post.sample(jax.random.PRNGKey(0), jnp.array([1.0, -1.0]))
    assert s.shape == (800, dim, 1)
    assert jnp.all(jnp.isfinite(s))
    # posterior ~ prior N(0, I)
    assert jnp.allclose(jnp.mean(s[..., 0], axis=0), 0.0, atol=0.25)


def test_gaussian_mock_matches_analytic_posterior():
    # likelihood N(x; theta, I), prior N(0, I)  =>  posterior N(x_o/2, 0.5 I)
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior, num_warmup=500, num_samples=3000)
    x_o = jnp.array([1.0, -1.0])

    s = post.sample(jax.random.PRNGKey(1), x_o)[..., 0]    # (3000, dim)
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.1)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.15)
