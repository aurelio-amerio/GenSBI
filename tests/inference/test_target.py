import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from gensbi.core.prior import make_gaussian_prior
from gensbi.inference.posterior import NLEPosterior, PosteriorTarget


class GaussianMock:
    """log q(x | theta) = sum_i N(x_i; theta_i, 1) (batched over rows)."""
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)   # (B,)


def test_build_target_decomposition_and_finiteness():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    target = post.build_target(jnp.array([1.0, -1.0]))

    assert isinstance(target, PosteriorTarget)
    assert target.dim == dim
    theta = jnp.array([0.3, 0.4])
    # log_posterior == log_prior + log_likelihood
    assert jnp.allclose(target.log_posterior(theta),
                        target.log_prior(theta) + target.log_likelihood(theta))
    # value and grad finite
    val = target.log_posterior(theta)
    grad = jax.grad(target.log_posterior)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (dim,) and jnp.all(jnp.isfinite(grad))


def test_log_likelihood_matches_flow():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    x_o = jnp.array([1.0, -1.0])
    theta = jnp.array([0.3, 0.4])
    target = post.build_target(x_o)
    expected = GaussianMock().log_prob(x_o[None], theta[None])[0]
    assert jnp.allclose(target.log_likelihood(theta), expected)


def test_structured_obs_keeps_observation_shape():
    # structured_obs: x_o is an image; theta stays a flat vector.
    dim = 2
    H = W = 4

    class ImageFlow:
        def log_prob(self, x, cond):
            # assert x retained its (B, H, W) structure
            assert x.shape == (1, H, W)
            return -0.5 * jnp.sum(cond ** 2, axis=-1)  # (B,)

    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(ImageFlow(), prior, structured_obs=True)
    x_o = jnp.ones((H, W))
    target = post.build_target(x_o)
    assert jnp.isfinite(target.log_likelihood(jnp.array([0.1, 0.2])))


def test_dim_reflects_prior_shape():
    prior = make_gaussian_prior((3,))
    post = NLEPosterior(GaussianMock(), prior)
    target = post.build_target(jnp.array([1.0, 2.0, 3.0]))
    assert target.dim == 3
