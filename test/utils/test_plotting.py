import os
os.environ['JAX_PLATFORMS']="cpu"

import numpy as np
import gensbi.utils.plotting as plotting

def test_plot_trajectories_runs():
    traj = np.random.randn(10, 5, 2)
    fig, ax = plotting.plot_trajectories(traj)
    assert fig is not None
    assert ax is not None

def test_plot_marginals_2d_runs():
    data = np.random.randn(100, 2)
    g = plotting.plot_marginals(data)
    assert g is not None

def test_plot_marginals_nd_runs():
    data = np.random.randn(100, 3)
    g = plotting.plot_marginals(data)
    assert g is not None
