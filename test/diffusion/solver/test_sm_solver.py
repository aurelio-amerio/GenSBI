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
# 3D shape tests: (batch, features, channels)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_3d(sde_cls):
    """Test sampling with input shape (batch, features, channels)."""
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
    assert samples.shape == (6, 5, 3, 4)  # (n_steps+1, batch, features, channels)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_channel1(sde_cls):
    """Test with channel=1 (user adds trailing dim to 2D data)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 1))

    samples = solver.sample(key, x_init, nsteps=5)
    assert samples.shape == (5, 3, 1)


# =========================================================
# 2D inputs should raise an error
# =========================================================


def test_sm_solver_rejects_2d_input():
    """2D inputs (batch, features) should raise AssertionError."""
    score_model = DummyScoreModel()
    sde = VPSmScheduler()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3))

    with pytest.raises(AssertionError, match="batch, features, channels"):
        solver.sample(key, x_init, nsteps=5)


# =========================================================
# CFG scale not implemented test
# =========================================================


def test_sm_solver_cfg_scale_not_implemented():
    sde = VPSmScheduler()
    path = SMPath(sde)
    solver = SMSolver(score_model=DummyScoreModel(), path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 2, 1))
    with pytest.raises(NotImplementedError):
        solver.sample(key, x_init, nsteps=2, cfg_scale=1.0)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_batch_independence(sde_cls):
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
    x_init = path.sample_prior(key, (n_batch, 3, 4))

    samples = solver.sample(key, x_init, nsteps=10)
    assert samples.shape == (n_batch, 3, 4)

    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"
