import jax.numpy as jnp

from gensbi.models.core.stats import fit_stat


def test_fit_stat_dim_vector():
    out = fit_stat(jnp.arange(3.0), (3, 2))
    assert out.shape == (3, 2)
    assert jnp.array_equal(out[:, 0], jnp.arange(3.0))
    assert jnp.array_equal(out[:, 0], out[:, 1])


def test_fit_stat_full_shape_passthrough():
    s = jnp.arange(6.0).reshape(3, 2)
    assert jnp.array_equal(fit_stat(s, (3, 2)), s)


def test_fit_stat_per_channel():
    out = fit_stat(jnp.array([1.0, 2.0]), (3, 2))   # (C,) with C != dim
    assert out.shape == (3, 2)
    assert jnp.array_equal(out[0], jnp.array([1.0, 2.0]))
    assert jnp.array_equal(out[0], out[1])


def test_fit_stat_scalar_and_dtype():
    out = fit_stat(1.5, (4, 1), dtype=jnp.float32)
    assert out.shape == (4, 1)
    assert out.dtype == jnp.float32
    assert jnp.all(out == 1.5)
