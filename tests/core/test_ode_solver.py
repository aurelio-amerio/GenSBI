"""
Tests for ``gensbi.core.ode_solver`` — ODESolver base class.

Covers:
- ``sample()`` convenience method
- ``get_log_prob()`` with fixed-step (Euler) solver
- ``get_log_prob()`` with Hutchinson divergence (missing key → ValueError)
- ``compute_log_prob()`` convenience method
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.flow_matching.solver.fm_ode_solver import FMODESolver
from gensbi.utils.model_wrapping import ModelWrapper


DIM = 4
CH = 1
BATCH_SIZE = 8
SHAPE = (BATCH_SIZE, DIM, CH)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def x_data(key):
    return jax.random.normal(key, SHAPE)


class DummyVelocityModel(nnx.Module):
    """Identity model: returns obs unchanged."""
    def __call__(self, obs, t, **kwargs):
        return obs


@pytest.fixture
def solver():
    model = DummyVelocityModel()
    wrapped = ModelWrapper(model)
    return FMODESolver(velocity_model=wrapped)


class TestODESolverSample:
    """Test the ``sample()`` convenience method (delegates to get_sampler)."""

    def test_sample_basic(self, solver, x_data):
        result = solver.sample(
            x_data,
            step_size=0.1,
            method="Euler",
            time_grid=jnp.array([0.0, 1.0]),
        )
        assert result.shape == SHAPE

    def test_sample_with_model_extras(self, solver, x_data):
        result = solver.sample(
            x_data,
            step_size=0.1,
            method="Euler",
            time_grid=jnp.array([0.0, 1.0]),
            model_extras={"dummy_key": jnp.ones(1)},
        )
        assert result.shape == SHAPE


class TestODESolverLogProb:
    """Test ``get_log_prob()`` and ``compute_log_prob()``."""

    def test_log_prob_euler_fixed_step(self, solver, x_data):
        """Cover the Euler branch in get_log_prob (fixed-step, not Dopri5)."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
            time_grid=jnp.array([1.0, 0.0]),
        )
        result = log_prob_fn(x_data)
        assert result.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(result))

    def test_log_prob_dopri5_adaptive(self, solver, x_data):
        """Cover the Dopri5 adaptive branch."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Dopri5",
            time_grid=jnp.array([1.0, 0.0]),
        )
        result = log_prob_fn(x_data)
        assert result.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(result))

    def test_log_prob_hutchinson_missing_key_raises(self, solver, x_data):
        """Hutchinson divergence without a key should raise ValueError."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
            time_grid=jnp.array([1.0, 0.0]),
            exact_divergence=False,
        )
        with pytest.raises(ValueError, match="PRNG key is required"):
            log_prob_fn(x_data)

    def test_log_prob_hutchinson_with_key(self, solver, x_data, key):
        """Hutchinson divergence with a key should work."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
            time_grid=jnp.array([1.0, 0.0]),
            exact_divergence=False,
        )
        result = log_prob_fn(x_data, key=key)
        assert result.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(result))

    def test_compute_log_prob_convenience(self, solver, x_data):
        """``compute_log_prob()`` delegates to ``get_log_prob()``."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        result = solver.compute_log_prob(
            x_data,
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
            time_grid=jnp.array([1.0, 0.0]),
        )
        assert result.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(result))


class TestODESolverDefaults:
    """Tests for default parameter branches (time_grid=None, static_model_kwargs=None)."""

    def test_get_sampler_default_time_grid(self, solver, x_data):
        """get_sampler with time_grid=None uses default [0, 1] (L107-108)."""
        sampler = solver.get_sampler(step_size=0.1, method="Euler")
        result = sampler(x_data)
        assert result.shape == SHAPE

    def test_get_log_prob_default_time_grid(self, solver, x_data):
        """get_log_prob with time_grid=None uses default [1, 0] (L236-237)."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
        )
        result = log_prob_fn(x_data)
        assert result.shape == (BATCH_SIZE,)

    def test_get_log_prob_default_static_kwargs(self, solver, x_data):
        """get_log_prob with static_model_kwargs=None (L242-243)."""
        log_p0 = lambda x: jnp.sum(
            jax.scipy.stats.norm.logpdf(x), axis=(-1, -2)
        )
        log_prob_fn = solver.get_log_prob(
            log_p0=log_p0,
            step_size=0.5,
            method="Euler",
            time_grid=jnp.array([1.0, 0.0]),
            static_model_kwargs=None,
        )
        result = log_prob_fn(x_data)
        assert result.shape == (BATCH_SIZE,)

    def test_get_sampler_with_diffrax_solver_instance(self, solver, x_data):
        """get_sampler with a diffrax solver instance (non-string method)."""
        import diffrax
        sampler = solver.get_sampler(
            step_size=0.1,
            method=diffrax.Euler(),
            time_grid=jnp.array([0.0, 1.0]),
        )
        result = sampler(x_data)
        assert result.shape == SHAPE
