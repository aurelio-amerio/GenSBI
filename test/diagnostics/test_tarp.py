# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest

from gensbi.diagnostics.tarp import check_tarp, run_tarp, plot_tarp, TARPResult
from gensbi.diagnostics.metrics import l1


def get_tarp_data():
    class DummyPipeline:
        def __init__(self):
            self.dim_obs = 3
            self.ch_obs = 2
            self.dim_cond = 4
            self.ch_cond = 3

    pipeline = DummyPipeline()
    # Create small dataset for testing
    num_tarp_samples = 20
    num_posterior_samples = 50
    dim_theta = pipeline.dim_obs * pipeline.ch_obs

    # xs = jax.random.normal(
    #     jax.random.PRNGKey(0), (num_tarp_samples, pipeline.dim_cond * pipeline.ch_cond)
    # )
    thetas = (
        jax.random.normal(jax.random.PRNGKey(1), (num_tarp_samples, dim_theta)) + 1.05
    )

    # run_tarp expects: (num_posterior_samples, num_tarp_samples, dim_theta)
    posterior_samples = (
        jax.random.normal(
            jax.random.PRNGKey(12345),
            (num_posterior_samples, num_tarp_samples, dim_theta),
        )
        + 1.05
    )

    return thetas, posterior_samples


def test_tarp_basic():
    thetas, posterior_samples = get_tarp_data()

    result = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
        bootstrap=False,
    )

    assert isinstance(result, TARPResult)
    # Check Jeffrey's bands are populated by default when bootstrap=False
    assert result.ecp_lower is not None
    assert result.ecp_upper is not None

    fig, ax = plot_tarp(result, mode="credibility")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_tarp_input_validation():
    thetas, posterior_samples = get_tarp_data()

    # Test wrong posterior samples shape
    bad_posterior = posterior_samples[:, :-1, :]  # modify num_tarp_samples dimension
    with pytest.raises(AssertionError, match="Wrong posterior samples shape"):
        run_tarp(thetas, bad_posterior)

    # Test wrong references shape (passed via _run_tarp implicitly if we pass references)
    references_bad = jax.random.normal(
        jax.random.PRNGKey(99), (thetas.shape[0] + 1, thetas.shape[1])
    )
    # The validation for references happens deep inside, or broadcasting fails.
    # run_tarp does not explicitly assert reference shape, but computation might fail.
    # Actually distances computation will fail with broadcasting error if shapes mismatch.
    try:
        run_tarp(thetas, posterior_samples, references=references_bad)
    except (ValueError, TypeError, AssertionError):
        pass  # Expected failure


def test_tarp_options():
    thetas, posterior_samples = get_tarp_data()

    # Test z_score options
    res1 = run_tarp(thetas, posterior_samples, z_score_theta=True)
    res2 = run_tarp(thetas, posterior_samples, z_score_theta=False)
    # Just check they run and return correct shapes, values might differ
    assert res1.ecp.shape == res2.ecp.shape

    # Test explicit num_bins
    res_bins = run_tarp(thetas, posterior_samples, num_bins=10)
    assert len(res_bins.alpha) == 11  # histogram edges = bins + 1

    # Test explicit references
    refs = jax.random.normal(jax.random.PRNGKey(42), thetas.shape)
    res_refs = run_tarp(thetas, posterior_samples, references=refs)

    # Test l1 distance
    res_l1 = run_tarp(thetas, posterior_samples, distance=l1)


def test_check_tarp():
    thetas, posterior_samples = get_tarp_data()
    result = run_tarp(thetas, posterior_samples)

    atc, ks_prob = check_tarp(result)

    assert isinstance(atc, float)
    assert isinstance(ks_prob, float)
    assert 0 <= ks_prob <= 1


def test_plot_tarp_options():
    thetas, posterior_samples = get_tarp_data()
    result = run_tarp(thetas, posterior_samples, bootstrap=False)

    # Test credibility plot
    fig, ax = plot_tarp(result, title="My TARP Plot", mode="credibility")
    assert ax.get_title() == "My TARP Plot"
    plt.close(fig)

    # Test confidence plot
    fig, ax = plot_tarp(result, title="Confidence Plot", mode="confidence")
    assert ax.get_title() == "Confidence Plot"
    plt.close(fig)

    # Test both
    fig, axes = plot_tarp(result, title="Both Plots", mode="both")
    assert len(axes) == 2
    plt.close(fig)


def test_tarp_bootstrap():
    thetas, posterior_samples = get_tarp_data()

    result = run_tarp(thetas, posterior_samples, bootstrap=True, num_bootstrap=10)

    assert isinstance(result, TARPResult)
    # Bootstrap should rely on bootstrap samples for bands, they are calculated in run_tarp
    assert result.ecp_lower is not None
    assert result.ecp_upper is not None
    assert result.ecp.shape[0] == 10  # num_bootstrap
