import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from numpyro import distributions as dist

from gensbi.flow_matching.solver.fm_sde_solver import NewZeroEndsSolver, NewNonSingularSolver
from gensbi.utils.model_wrapping import ModelWrapper


class DummyModel(nnx.Module):
    """Dummy velocity model that returns zeros (same shape as input)."""

    def __call__(self, obs, t, *args, **kwargs):
        return jnp.zeros_like(obs)


def make_solver(solver_cls, features=3, channels=2, alpha=0.5, eps0=1e-3):
    """Helper to create a solver with mu0 of shape (features, channels)."""
    model = DummyModel()
    wrapper = ModelWrapper(model)
    mu0 = jnp.zeros((features, channels))
    sigma0 = jnp.ones((features, channels))
    if solver_cls == NewNonSingularSolver:
        return solver_cls(wrapper, mu0, sigma0, alpha=alpha)
    else:
        return solver_cls(wrapper, mu0, sigma0, alpha=alpha, eps0=eps0)


def _sample_prior(mu0, sigma0, key, nsamples):
    """Sample from Normal(mu0, sigma0) and return (nsamples, *mu0.shape)."""
    p = dist.Independent(
        dist.Normal(loc=mu0, scale=sigma0),
        reinterpreted_batch_ndims=mu0.ndim,
    )
    return p.sample(key, (nsamples,))


# =========================================================
# Initialization tests
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_initialization(solver_cls):
    solver = make_solver(solver_cls)
    assert solver.mu0.shape == (3, 2)
    assert solver.sigma0.shape == (3, 2)
    assert solver.alpha == 0.5


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_attributes(solver_cls):
    """Verify key attributes are stored correctly."""
    solver = make_solver(solver_cls, features=4, channels=3, alpha=0.8)
    assert solver.mu0.shape == (4, 3)
    assert solver.sigma0.shape == (4, 3)
    assert solver.alpha == 0.8


# =========================================================
# 3D shape tests: output (nsamples, features, channels)
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_sample_3d(solver_cls):
    solver = make_solver(solver_cls, features=3, channels=4)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 5
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    samples = solver.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_sample_channel1(solver_cls):
    """Channel=1 case (user with effectively 1D features)."""
    solver = make_solver(solver_cls, features=4, channels=1)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 5
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    samples = solver.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
    assert samples.shape == (5, 4, 1)


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_sample_intermediates(solver_cls):
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 5
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    time_grid = jnp.linspace(0, 1, 6)
    samples = solver.sample(
        x_init, step_size=0.2, method="Euler", return_intermediates=True,
        time_grid=time_grid, key=key_sample,
    )
    assert samples.shape == (6, 5, 3, 2)  # (n_steps+1, nsamples, features, channels)


# =========================================================
# get_sampler API test
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_get_sampler_api(solver_cls):
    solver = make_solver(solver_cls)
    sampler = solver.get_sampler(step_size=0.2, method="Euler")
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 4
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    samples = sampler(x_init, key_sample)
    assert samples.shape == (4, 3, 2)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_batch_independence(solver_cls):
    """Verify all samples in a batch are independent (not identical)."""
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(42)
    key_init, key_sample = jax.random.split(key)

    n_batch = 8
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, n_batch)

    # Needs enough steps to diverge if there is noise
    samples = solver.sample(x_init, step_size=0.1, method="Euler", key=key_sample)
    assert samples.shape == (n_batch, 3, 2)

    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"


# =========================================================
# Method validation test
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_invalid_method(solver_cls):
    solver = make_solver(solver_cls)
    with pytest.raises(ValueError, match="not supported"):
        solver.get_sampler(step_size=0.1, method="InvalidMethod")


# =========================================================
# Coverage improvement tests
# =========================================================


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_custom_solver_instance(solver_cls):
    """Pass a diffrax solver instance instead of a string."""
    import diffrax

    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 3
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    # Pass Euler instance directly instead of string
    samples = solver.sample(
        x_init, step_size=0.2, method=diffrax.Euler(), key=key_sample
    )
    assert samples.shape == (3, 3, 2)


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_shark_method(solver_cls):
    """Test with ShARK method (adaptive step sizing via PIDController)."""
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 3
    x_init = _sample_prior(solver.mu0, solver.sigma0, key_init, nsamples)

    samples = solver.sample(
        x_init, step_size=0.2, method="ShARK", key=key_sample
    )
    assert samples.shape == (3, 3, 2)


@pytest.mark.parametrize("solver_cls", [NewZeroEndsSolver, NewNonSingularSolver])
def test_sample_key_none_error(solver_cls):
    """Calling sample with key=None should raise ValueError."""
    solver = make_solver(solver_cls)
    key = jax.random.PRNGKey(0)
    nsamples = 2
    x_init = _sample_prior(solver.mu0, solver.sigma0, key, nsamples)

    with pytest.raises(ValueError, match="key is required"):
        solver.sample(x_init, step_size=0.2, method="Euler", key=None)
