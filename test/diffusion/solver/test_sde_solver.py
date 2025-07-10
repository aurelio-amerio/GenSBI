import os
os.environ['JAX_PLATFORMS'] = "cpu"

import jax
import pytest
from gensbi.diffusion.solver.sde_solver import SDESolver
from gensbi.diffusion.path.edm_path import EDMPath
from gensbi.diffusion.path.scheduler.edm import EDMScheduler


def test_sde_solver_initialization():
    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=None, path=path)
    assert isinstance(solver, SDESolver)
    assert solver.path is path


def test_sde_solver_sample_shape():
    path = EDMPath(scheduler=EDMScheduler())
    solver = SDESolver(score_model=None, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (10, 2))
    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True)
    assert samples.shape[1:] == (10, 2)
