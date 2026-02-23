# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import matplotlib

from jax import numpy as jnp

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from gensbi.utils.plotting import (
    _parse_range,
    plot_trajectories,
    _plot_marginals_seaborn,
    plot_marginals,
    plot_2d_levels,
    plot_2d_dist_contour,
)

from numpyro import distributions as dist


# %%
def test_plot_trajectories_runs():
    traj = np.random.randn(10, 5, 2)
    fig, ax = plot_trajectories(traj)
    assert fig is not None
    assert ax is not None


def test_parse_range():
    ndim = 3
    range_arg = None
    parsed = _parse_range(range_arg, ndim)
    assert parsed == [None, None, None]

    range_arg = (-1, 1)
    parsed = _parse_range(range_arg, ndim)
    assert parsed == [(-1, 1), (-1, 1), (-1, 1)]

    range_arg = [(-1, 1), (0, 2), (3, 4)]
    parsed = _parse_range(range_arg, ndim)
    assert parsed == [(-1, 1), (0, 2), (3, 4)]

    with pytest.raises(ValueError) as e:
        _parse_range((1, 2, 3), ndim)
    assert (
        str(e.value)
        == "Range must be None, a tuple (min, max), or a sequence of such tuples, one per axis"
    )


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_nd(ndim):
    data = np.random.normal(size=(100, ndim))
    true_param = np.random.normal(size=(ndim,))
    # Should not raise

    plot_marginals(
        data,
        backend="seaborn",
    )

    plot_marginals(data, backend="seaborn", true_param=true_param)

    plot_marginals(
        data,
        backend="corner",
    )

    plot_marginals(data, backend="corner", true_param=true_param)

    plot_marginals(
        data,
        backend="seaborn",
        plot_levels=False,
    )

    with pytest.raises(ValueError) as e:
        plot_marginals(data, backend="unknown")
    assert str(e.value) == f"Unknown backend: unknown. Use 'corner' or 'seaborn'."

    plt.close("all")


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_plot_marginals_with_range(ndim):
    data = np.random.normal(size=(100, ndim))
    ranges = [(-2, 2)] * ndim

    plot_marginals(
        data,
        range=ranges,
        backend="seaborn",
    )

    plot_marginals(
        data,
        range=ranges,
        backend="corner",
    )

    plt.close("all")


def test_plot_marginals_labels():
    data = np.random.normal(size=(100, 3))
    labels = ["A", "B", "C"]

    plot_marginals(data, backend="seaborn", labels=labels)

    plot_marginals(data, backend="corner", labels=labels)

    plt.close("all")


def test_plot_marginals_invalid_range():
    data = np.random.normal(size=(100, 2))
    with pytest.raises(ValueError) as e:

        plot_marginals(data, backend="seaborn", range=[(-2, 2)])

    assert (
        str(e.value)
        == "Range must be None, a tuple (min, max), or a sequence of such tuples, one per axis"
    )

    with pytest.raises(ValueError) as e:

        plot_marginals(data, backend="corner", range=[(-2, 2)])

    assert (
        str(e.value)
        == "Range must be None, a tuple (min, max), or a sequence of such tuples, one per axis"
    )

    plt.close("all")


def test_plot_2d_levels():
    p0 = dist.Independent(
        dist.Normal(loc=jnp.zeros((2,)), scale=jnp.ones((2,))),
        reinterpreted_batch_ndims=1,
    )
    x = jnp.linspace(-3, 3, 100)
    y = jnp.linspace(-3, 3, 100)
    XY = jnp.meshgrid(x, y)
    XY = jnp.stack([XY[0].flatten(), XY[1].flatten()], axis=1)
    Z = p0.log_prob(XY)
    Z = np.exp(Z.reshape(100, 100))

    fig, ax = plt.subplots()
    plot_2d_levels(np.array(x), np.array(y), np.array(Z), ax=ax)
    plt.close("all")
    return


def test_plot_2d_dist_contour():
    p0 = dist.Independent(
        dist.Normal(loc=jnp.zeros((2,)), scale=jnp.ones((2,))),
        reinterpreted_batch_ndims=1,
    )
    true_param = jnp.array([0.0, 0.0])
    x = jnp.linspace(-3, 3, 100)
    y = jnp.linspace(-3, 3, 100)
    XY = jnp.meshgrid(x, y)
    XY = jnp.stack([XY[0].flatten(), XY[1].flatten()], axis=1)
    Z = p0.log_prob(XY)
    Z = np.exp(Z.reshape(100, 100))

    plot_2d_dist_contour(np.array(x), np.array(y), np.array(Z), true_param=true_param)
    plt.close("all")
    return
