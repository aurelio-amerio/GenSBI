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


def _rand_params(key, spline):
    return jax.random.normal(key, (spline.num_params,))


def test_scalar_roundtrip_inside_interval():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(1), spline)
    xs = jnp.linspace(-4.5, 4.5, 50)
    for x in xs:
        u, _ = spline._fwd_scalar(x, params)
        x_rec, _ = spline._inv_scalar(u, params)
        assert jnp.allclose(x_rec, x, atol=1e-4), (x, x_rec)


def test_scalar_logdet_matches_autodiff():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(2), spline)
    for x in jnp.linspace(-4.0, 4.0, 25):
        _, logderiv = spline._fwd_scalar(x, params)
        g = jax.grad(lambda z: spline._fwd_scalar(z, params)[0])(x)
        assert jnp.allclose(logderiv, jnp.log(jnp.abs(g)), atol=1e-4), (x, logderiv, g)


def test_tails_are_identity():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(3), spline)
    for x in [-8.0, 7.5]:
        u, logderiv = spline._fwd_scalar(jnp.array(x), params)
        assert jnp.allclose(u, x)              # identity outside [-B, B]
        assert jnp.allclose(logderiv, 0.0)


def test_vector_inverse_forward_roundtrip():
    spline = RQSpline(num_bins=6, range_bound=4.0)
    dim = 4
    key = jax.random.PRNGKey(4)
    kp, kx = jax.random.split(key)
    params = jax.random.normal(kp, (dim, spline.num_params))
    x = jax.random.uniform(kx, (dim,), minval=-3.5, maxval=3.5)
    u, ld_inv = spline.inverse(x, params)
    x_rec, ld_fwd = spline.forward(u, params)
    assert jnp.allclose(x_rec, x, atol=1e-4)
    assert jnp.allclose(ld_inv + ld_fwd, 0.0, atol=1e-4)   # logdets cancel


def test_rqspline_exported():
    from gensbi.normalizing_flows.bijections import RQSpline as A
    from gensbi.normalizing_flows import RQSpline as B
    assert A is RQSpline and B is RQSpline
