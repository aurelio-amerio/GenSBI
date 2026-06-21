import jax
import jax.numpy as jnp
import pytest

from gensbi.core.time_sampling import sample_time


def test_uniform_is_bit_identical_to_jax_uniform():
    key = jax.random.PRNGKey(0)
    n = 257
    got = sample_time(key, n, dist="uniform")
    expected = jax.random.uniform(key, (n,))
    assert got.shape == (n,)
    assert jnp.array_equal(got, expected)


def test_logitnormal_in_unit_interval_and_deterministic():
    key = jax.random.PRNGKey(1)
    n = 100_000
    t = sample_time(key, n, dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    assert t.shape == (n,)
    assert bool(jnp.all(t > 0.0)) and bool(jnp.all(t < 1.0))
    t2 = sample_time(key, n, dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    assert jnp.array_equal(t, t2)  # deterministic for a fixed key


def test_logitnormal_logit_is_normal_m_s():
    # logit(t) = log(t) - log(1-t) must be Normal(mean, std) by construction.
    key = jax.random.PRNGKey(2)
    n = 200_000
    m, s = 0.5, 1.3
    t = sample_time(key, n, dist="logitnormal", logitnorm_mean=m, logitnorm_std=s)
    z = jnp.log(t) - jnp.log1p(-t)
    assert abs(float(jnp.mean(z)) - m) < 0.02
    assert abs(float(jnp.std(z)) - s) < 0.02


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        sample_time(jax.random.PRNGKey(0), 8, dist="cosmap")


def _build_method(**kw):
    from gensbi.core import FlowMatchingMethod
    method = FlowMatchingMethod(**kw)
    method.build_path(config={}, event_shape=(4, 1))  # sets method.prior
    return method


def test_method_default_prepare_batch_bit_identical():
    method = _build_method()
    key = jax.random.PRNGKey(7)
    x1 = jnp.zeros((16, 4, 1))
    _, _, t = method.prepare_batch(key, x1, path=None)
    # reproduce the exact historical computation (split, then uniform on 2nd sub-key)
    _, rng_t = jax.random.split(key)
    expected_t = jax.random.uniform(rng_t, (16,))
    assert jnp.array_equal(t, expected_t)


def test_method_logitnormal_prepare_batch():
    method = _build_method(time_dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    key = jax.random.PRNGKey(7)
    x1 = jnp.zeros((4096, 4, 1))
    _, _, t = method.prepare_batch(key, x1, path=None)
    assert bool(jnp.all(t > 0)) and bool(jnp.all(t < 1))
    _, rng_t = jax.random.split(key)
    assert jnp.array_equal(t, sample_time(rng_t, 4096, dist="logitnormal"))


def test_method_logitnormal_custom_params_flow_through():
    m, s = 2.0, 0.5
    method = _build_method(time_dist="logitnormal", logitnorm_mean=m, logitnorm_std=s)
    key = jax.random.PRNGKey(7)
    x1 = jnp.zeros((4096, 4, 1))
    _, _, t = method.prepare_batch(key, x1, path=None)
    _, rng_t = jax.random.split(key)
    expected = sample_time(rng_t, 4096, dist="logitnormal", logitnorm_mean=m, logitnorm_std=s)
    assert jnp.array_equal(t, expected)


def test_method_invalid_time_dist_raises():
    from gensbi.core import FlowMatchingMethod
    with pytest.raises(ValueError):
        FlowMatchingMethod(time_dist="bogus")
