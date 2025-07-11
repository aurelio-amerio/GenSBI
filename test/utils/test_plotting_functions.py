import os
os.environ['JAX_PLATFORMS'] = "cpu"
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from gensbi.utils.plotting import _plot_marginals_2d, _plot_marginals_nd, plot_marginals, plot_trajectories

def test_plot_trajectories_runs():
    traj = np.random.randn(10, 5, 2)
    fig, ax = plot_trajectories(traj)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_nd(ndim):
    data = np.random.normal(size=(100, ndim))
    # Should not raise
    fig, axes = _plot_marginals_nd(data)
    plt.close(fig)
    if ndim == 2:
        g = _plot_marginals_2d(data)
        plt.close(g.figure)
    g3 = plot_marginals(data)
    # plot_marginals returns PairGrid for ndim==2, (fig, axes) for ndim>2
    if ndim == 2:
        plt.close(g3.figure)
    else:
        fig, axes = g3
        plt.close(fig)

@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_with_range(ndim):
    data = np.random.normal(size=(100, ndim))
    ranges = [(-2, 2)] * ndim
    fig, axes = _plot_marginals_nd(data, range=ranges)
    plt.close(fig)
    if ndim == 2:
        g = _plot_marginals_2d(data, range=ranges)
        plt.close(g.figure)
    g3 = plot_marginals(data, range=ranges)
    if ndim == 2:
        plt.close(g3.figure)
    else:
        fig, axes = g3
        plt.close(fig)

def test_plot_marginals_labels():
    data = np.random.normal(size=(100, 3))
    labels = ["A", "B", "C"]
    fig, axes = _plot_marginals_nd(data, labels=labels)
    plt.close(fig)
    g2 = _plot_marginals_2d(data[:, :2], labels=labels[:2])
    plt.close(g2.figure)
    g3 = plot_marginals(data, labels=labels)
    if isinstance(g3, tuple):
        fig, axes = g3
        plt.close(fig)
    else:
        plt.close(g3.figure)

def test_plot_marginals_invalid_range():
    data = np.random.normal(size=(100, 2))
    with pytest.raises(ValueError):
        _plot_marginals_2d(data, range=[(-2, 2)])
    with pytest.raises(ValueError):
        _plot_marginals_nd(data, range=[(-2, 2)])
    with pytest.raises(ValueError):
        plot_marginals(data, range=[(-2, 2)])
