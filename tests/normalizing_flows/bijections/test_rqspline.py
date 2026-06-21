import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.normalizing_flows.bijections.transformers import RQSpline


def test_num_params():
    assert RQSpline(num_bins=8).num_params == 3 * 8 - 1
    assert RQSpline(num_bins=4).num_params == 3 * 4 - 1


def test_knots_constraints():
    K, B = 8, 5.0
    spline = RQSpline(num_bins=K, range_bound=B)
    params = jax.random.normal(jax.random.PRNGKey(0), (spline.num_params,))
    x_knots, y_knots, d = spline._knots(params)

    assert x_knots.shape == (K + 1,)
    assert y_knots.shape == (K + 1,)
    assert d.shape == (K + 1,)
    # span exactly [-B, B], strictly increasing, positive derivatives
    assert jnp.allclose(x_knots[0], -B) and jnp.allclose(x_knots[-1], B)
    assert jnp.allclose(y_knots[0], -B) and jnp.allclose(y_knots[-1], B)
    assert jnp.all(jnp.diff(x_knots) > 0)
    assert jnp.all(jnp.diff(y_knots) > 0)
    assert jnp.all(d > 0)
    # linear tails: boundary derivatives are 1
    assert jnp.allclose(d[0], 1.0) and jnp.allclose(d[-1], 1.0)


def test_zero_params_give_identity_knots():
    K, B = 8, 5.0
    spline = RQSpline(num_bins=K, range_bound=B)
    x_knots, y_knots, d = spline._knots(jnp.zeros(spline.num_params))
    # uniform bins => x_knots == y_knots, and all derivatives == 1
    assert jnp.allclose(x_knots, y_knots, atol=1e-5)
    assert jnp.allclose(d, jnp.ones(K + 1), atol=1e-5)
