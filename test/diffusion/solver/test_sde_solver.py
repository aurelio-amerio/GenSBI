import os
os.environ['JAX_PLATFORMS'] = "cpu"

import jax
import pytest
from gensbi.diffusion.solver import SDESolver
from gensbi.diffusion.path.edm_path import EDMPath
from gensbi.diffusion.path.scheduler.edm import EDMScheduler, VEScheduler, VPScheduler

from flax import nnx


class DummyScoreModel(nnx.Module):
    def __call__(self, obs, t):
        return jax.numpy.zeros_like(obs)
    


@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sde_solver_initialization(scheduler_cls):
    scheduler = scheduler_cls()
    path = EDMPath(scheduler=scheduler)
    solver = SDESolver(score_model=None, path=path)
    assert isinstance(solver, SDESolver)
    assert solver.path is path


@pytest.mark.parametrize("scheduler_cls", [EDMScheduler, VEScheduler, VPScheduler])
def test_sde_solver_sample_shape(scheduler_cls):
    score_model = DummyScoreModel()
    scheduler = scheduler_cls()
    path = EDMPath(scheduler=scheduler)
    solver = SDESolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (10, 2))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="Heun")
    assert samples.shape[1:] == (10, 2)

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="Euler")
    assert samples.shape[1:] == (10, 2)

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=False, method="Heun")
    assert samples.shape == (10, 2)

    # test error if we use a method that is not implemented
    with pytest.raises(AssertionError) as e:
        solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="RK4")
    assert "Unknown method" in str(e.value)

    with pytest.raises(AssertionError) as e:
        solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="Heun", condition_mask=1)
    assert "Condition value must be provided if condition mask is provided" in str(e.value)


def test_sde_solver_cfg_scale_not_implemented():
    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=DummyScoreModel(), path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 2))
    with pytest.raises(NotImplementedError) as e:
        solver.sample(key, x_init, nsteps=2, cfg_scale=1.0)
    assert "CFG scale is not implemented" in str(e.value)
