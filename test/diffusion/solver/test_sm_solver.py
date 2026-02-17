import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.diffusion.solver.sm_solver import SMSolver
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


class DummyScoreModel(nnx.Module):
    def __call__(self, obs, t, **kwargs):
        return jnp.zeros_like(obs)


# =========================================================
# Basic initialization tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_initialization(sde_cls):
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=None, path=path)
    assert isinstance(solver, SMSolver)
    assert solver.path is path


# =========================================================
# 2D shape tests: (batch, dim)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_2d(sde_cls):
    """Test sampling with input shape (batch, dim)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 2))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=False)
    assert samples.shape == (5, 2)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_2d_intermediates(sde_cls):
    """Test sampling with intermediates for 2D input."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 2))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True)
    assert samples.shape == (6, 5, 2)  # (n_steps+1, batch, dim)


# =========================================================
# 3D shape tests: (batch, features, channel)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_3d(sde_cls):
    """Test sampling with input shape (batch, features, channel)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=False)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_3d_intermediates(sde_cls):
    """Test sampling with intermediates for 3D input."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True)
    assert samples.shape == (6, 5, 3, 4)  # (n_steps+1, batch, features, channel)


# =========================================================
# CFG scale not implemented test
# =========================================================


def test_sm_solver_cfg_scale_not_implemented():
    sde = VPSmScheduler()
    path = SMPath(sde)
    solver = SMSolver(score_model=DummyScoreModel(), path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 2))
    with pytest.raises(NotImplementedError):
        solver.sample(key, x_init, nsteps=2, cfg_scale=1.0)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_batch_independence_2d(sde_cls):
    """Verify all samples in a batch are independent (not identical).

    Regression test: a previous bug caused all samples to share the same
    Brownian motion, producing identical outputs.
    """
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(42)
    n_batch = 8
    x_init = path.sample_prior(key, (n_batch, 3))

    samples = solver.sample(key, x_init, nsteps=10)
    assert samples.shape == (n_batch, 3)

    # Check that no two samples are identical
    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_batch_independence_3d(sde_cls):
    """Same independence check for (batch, features, channel) shapes."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(42)
    n_batch = 8
    x_init = path.sample_prior(key, (n_batch, 3, 4))

    samples = solver.sample(key, x_init, nsteps=10)
    assert samples.shape == (n_batch, 3, 4)

    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"
