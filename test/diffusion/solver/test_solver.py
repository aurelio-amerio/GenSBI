import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.solver.solver import Solver


def test_solver_initialization():
    solver = Solver()
    assert isinstance(solver, Solver)
