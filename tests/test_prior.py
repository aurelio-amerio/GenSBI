import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.core.prior import make_gaussian_prior, is_gaussian_prior


def test_make_gaussian_prior_shape():
    prior = make_gaussian_prior(5, 2)
    samples = prior.sample(jax.random.PRNGKey(0), (10,))
    assert samples.shape == (10, 5, 2)


def test_make_gaussian_prior_log_prob():
    prior = make_gaussian_prior(3, 1)
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 1))
    lp = prior.log_prob(x)
    assert lp.shape == (4,)


def test_make_gaussian_prior_custom_params():
    prior = make_gaussian_prior(2, 1, mu=1.0, sigma=2.0)
    assert jnp.allclose(prior.base_dist.loc, jnp.full((2, 1), 1.0))
    assert jnp.allclose(prior.base_dist.scale, jnp.full((2, 1), 2.0))


def test_is_gaussian_prior():
    prior = make_gaussian_prior(3, 1)
    assert is_gaussian_prior(prior)


def test_is_gaussian_prior_false():
    import numpyro.distributions as dist

    prior = dist.Uniform(0, 1)
    assert not is_gaussian_prior(prior)


def test_legacy_two_int_form():
    prior = make_gaussian_prior(5, 1)
    assert prior.event_shape == (5, 1)
    assert is_gaussian_prior(prior)


def test_event_shape_tuple_form():
    prior = make_gaussian_prior((8, 8, 2))
    assert prior.event_shape == (8, 8, 2)
    assert is_gaussian_prior(prior)
    s = prior.sample(jax.random.PRNGKey(0), (4,))
    assert s.shape == (4, 8, 8, 2)


def test_three_positional_ints_raise():
    """make_gaussian_prior(H, W, C) used to silently read C as the MEAN."""
    with pytest.raises(TypeError):
        make_gaussian_prior(8, 8, 2)


def test_mu_sigma_keywords():
    prior = make_gaussian_prior((4, 4, 1), mu=2.0, sigma=3.0)
    assert jnp.allclose(prior.base_dist.loc, 2.0)
    assert jnp.allclose(prior.base_dist.scale, 3.0)
