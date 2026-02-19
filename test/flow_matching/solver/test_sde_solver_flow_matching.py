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
    samples = solver.sample(key, nsamples=5, nsteps=5, method="Euler")
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_channel1(solver_cls):
    """Channel=1 case (user with effectively 1D features)."""
    solver = make_solver(solver_cls, features=4, channels=1)
    key = jax.random.PRNGKey(0)
    samples = solver.sample(key, nsamples=5, nsteps=5, method="Euler")
    assert samples.shape == (5, 4, 1)


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_sample_intermediates(solver_cls):
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(0)
    samples = solver.sample(
        key, nsamples=5, nsteps=5, method="Euler", return_intermediates=True
    )
    assert samples.shape == (6, 5, 3, 2)  # (n_steps+1, nsamples, features, channels)


# =========================================================
# get_sampler API test
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_get_sampler_api(solver_cls):
    solver = make_solver(solver_cls)
    sampler = solver.get_sampler(nsteps=5, method="Euler")
    key = jax.random.PRNGKey(0)
    samples = sampler(key, 4)
    assert samples.shape == (4, 3, 2)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("solver_cls", [ZeroEnds, NonSingular])
def test_batch_independence(solver_cls):
    """Verify all samples in a batch are independent (not identical)."""
    solver = make_solver(solver_cls, features=3, channels=2)
    key = jax.random.PRNGKey(42)
    n_batch = 8
    samples = solver.sample(key, nsamples=n_batch, nsteps=10, method="Euler")
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
        solver.get_sampler(nsteps=5, method="InvalidMethod")
