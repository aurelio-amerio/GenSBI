import os
os.environ['JAX_PLATFORMS'] = "cpu"

import jax
import pytest
from gensbi.diffusion.solver import SDESolver
from gensbi.diffusion.path.edm_path import EDMPath
from gensbi.diffusion.path.scheduler.edm import EDMScheduler

from flax import nnx


def test_sde_solver_initialization():
    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=None, path=path)
    assert isinstance(solver, SDESolver)
    assert solver.path is path


def test_sde_solver_sample_shape():

    class DummyScoreModel(nnx.Module):
        def __call__(self, x, t):
            return jax.numpy.zeros_like(x)
        
    score_model = DummyScoreModel()

    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (10, 2))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="Heun")
    assert samples.shape[1:] == (10, 2)

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True, method="Euler")
    assert samples.shape[1:] == (10, 2)


def test_sde_solver_cfg_scale_not_implemented():
    class DummyScoreModel:
        def __call__(self, x, t):
            return x
    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=DummyScoreModel(), path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 2))
    try:
        solver.sample(key, x_init, nsteps=2, cfg_scale=1.0)
    except NotImplementedError as e:
        assert "CFG scale is not implemented" in str(e)
