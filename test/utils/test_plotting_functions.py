import os

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from gensbi.utils.plotting import (
    _plot_marginals_2d,
    _plot_marginals_nd,
    plot_marginals,
    plot_trajectories,
)


def test_plot_trajectories_runs():
    traj = np.random.randn(10, 5, 2)
    fig, ax = plot_trajectories(traj)
    assert fig is not None
    assert ax is not None


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_nd(ndim):
    data = np.random.normal(size=(100, ndim))
    # Should not raise
    _plot_marginals_nd(data)
    plt.clf()
    if ndim == 2:
        _plot_marginals_2d(data)
        plt.clf()
    plot_marginals(data)
    plt.clf()


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_with_range(ndim):
    data = np.random.normal(size=(100, ndim))
    ranges = [(-2, 2)] * ndim
    _plot_marginals_nd(data, range=ranges)
    plt.clf()
    if ndim == 2:
        _plot_marginals_2d(data, range=ranges)
        plt.clf()
    _plot_marginals_nd(data, range=ranges)
    plt.clf()


def test_plot_marginals_labels():
    data = np.random.normal(size=(100, 3))
    labels = ["A", "B", "C"]
    _plot_marginals_nd(data, labels=labels)
    plt.clf()
    _plot_marginals_2d(data[:, :2], labels=labels[:2])
    plt.clf()
    plot_marginals(data, labels=labels)
    plt.clf()


def test_plot_marginals_invalid_range():
    data = np.random.normal(size=(100, 2))
    with pytest.raises(ValueError):
        _plot_marginals_2d(data, range=[(-2, 2)])
    with pytest.raises(ValueError):
        _plot_marginals_nd(data, range=[(-2, 2)])
    with pytest.raises(ValueError):
        plot_marginals(data, range=[(-2, 2)])
