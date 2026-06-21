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
