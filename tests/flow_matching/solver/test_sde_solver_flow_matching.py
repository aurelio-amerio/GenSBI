import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.flow_matching.solver.sde_solver_fm import ZeroEnds, NonSingular
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
    if solver_cls == NonSingular:
        return solver_cls(wrapper, mu0, sigma0, alpha=alpha)
    else:
        return solver_cls(wrapper, mu0, sigma0, alpha=alpha, eps0=eps0)


# =========================================================
# Initialization tests
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_initialization(solver_cls):
    solver = make_solver(solver_cls)
    assert solver.flat_dim == 6  # 3 * 2
    assert solver.sample_shape == (3, 2)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_rejects_1d_mu0(solver_cls):
    """1D mu0 should raise AssertionError."""
    model = DummyModel()
    wrapper = ModelWrapper(model)
    mu0 = jnp.zeros(3)  # 1D — not allowed
    sigma0 = jnp.ones(3)
    with pytest.raises(AssertionError, match="features, channels"):
        if solver_cls == NonSingular:
            solver_cls(wrapper, mu0, sigma0, alpha=0.5)
        else:
            solver_cls(wrapper, mu0, sigma0, alpha=0.5, eps0=1e-3)



# =========================================================
# 3D shape tests: output (nsamples, features, channels)
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_3d(solver_cls):
    solver = make_solver(solver_cls, features=3, channels=4)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    
    nsamples = 5
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    # Reshape to (nsamples, features, channels)
    x_init = x_init.reshape(nsamples, 3, 4)
    
    samples = solver.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_channel1(solver_cls):
    """Channel=1 case (user with effectively 1D features)."""
    solver = make_solver(solver_cls, features=4, channels=1)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    
    nsamples = 5
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    x_init = x_init.reshape(nsamples, 4, 1)

    samples = solver.sample(x_init, step_size=0.2, method="Euler", key=key_sample)
    assert samples.shape == (5, 4, 1)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_intermediates(solver_cls):
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    
    nsamples = 5
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    x_init = x_init.reshape(nsamples, 3, 2)

    samples = solver.sample(
        x_init, step_size=0.2, method="Euler", return_intermediates=True, key=key_sample
    )
    # nsteps = 1.0 / 0.2 = 5 steps. +1 for initial state? 
    # diffrax SaveAt(ts=time_grid). time_grid defaults to [0, 1].
    # If we want intermediate steps matching nsteps=5, we need time_grid to have 6 points.
    # The default time_grid is [0, 1], so it only saves at 0 and 1 if return_intermediates=True?
    # Wait, ODESolver logic:
    # saveat=diffrax.SaveAt(ts=time_grid).
    # If time_grid is [0, 1], it returns 2 frames.
    # If we want 6 frames, we must pass time_grid of length 6.
    
    time_grid = jnp.linspace(0, 1, 6)
    samples = solver.sample(
        x_init, step_size=0.2, method="Euler", return_intermediates=True, time_grid=time_grid, key=key_sample
    )
    assert samples.shape == (6, 5, 3, 2)  # (n_steps+1, nsamples, features, channels)


# =========================================================
# get_sampler API test
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_get_sampler_api(solver_cls):
    solver = make_solver(solver_cls)
    sampler = solver.get_sampler(step_size=0.2, method="Euler")
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)
    
    nsamples = 4
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    x_init = x_init.reshape(nsamples, 3, 2)

    samples = sampler(x_init, key_sample)
    assert samples.shape == (4, 3, 2)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_batch_independence(solver_cls):
    """Verify all samples in a batch are independent (not identical)."""
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(42)
    key_init, key_sample = jax.random.split(key)

    n_batch = 8
    x_init = solver.prior_distribution.sample(key_init, (n_batch,))
    x_init = x_init.reshape(n_batch, 3, 2)
    
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


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_invalid_method(solver_cls):
    solver = make_solver(solver_cls)
    with pytest.raises(ValueError, match="not supported"):
        solver.get_sampler(step_size=0.1, method="InvalidMethod")


# =========================================================
# Coverage improvement tests
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_custom_solver_instance(solver_cls):
    """Pass a diffrax solver instance instead of a string."""
    import diffrax

    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 3
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    x_init = x_init.reshape(nsamples, 3, 2)

    # Pass Euler instance directly instead of string
    samples = solver.sample(
        x_init, step_size=0.2, method=diffrax.Euler(), key=key_sample
    )
    assert samples.shape == (3, 3, 2)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_shark_method(solver_cls):
    """Test with ShARK method (adaptive step sizing via PIDController)."""
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    key_init, key_sample = jax.random.split(key)

    nsamples = 3
    x_init = solver.prior_distribution.sample(key_init, (nsamples,))
    x_init = x_init.reshape(nsamples, 3, 2)

    samples = solver.sample(
        x_init, step_size=0.2, method="ShARK", key=key_sample
    )
    assert samples.shape == (3, 3, 2)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_key_none_error(solver_cls):
    """Calling sample with key=None should raise ValueError."""
    solver = make_solver(solver_cls)
    key = jax.random.PRNGKey(0)
    x_init = solver.prior_distribution.sample(key, (2,))
    x_init = x_init.reshape(2, 3, 2)

    with pytest.raises(ValueError, match="key is required"):
        solver.sample(x_init, step_size=0.2, method="Euler", key=None)

