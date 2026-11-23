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
    with pytest.raises(ValueError, match="Wrong posterior samples shape"):
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


def test_tarp_num_bins_none_uses_one_bin_per_ten_samples():
    thetas, posterior_samples = get_tarp_data()  # 20 TARP samples -> 2 bins

    result = run_tarp(thetas, posterior_samples, num_bins=None)
    assert result.alpha.shape == (3,)
    assert result.ecp.shape == (3,)

    result_boot = run_tarp(
        thetas, posterior_samples, num_bins=None, bootstrap=True, num_bootstrap=4
    )
    assert result_boot.alpha.shape == (3,)
    assert result_boot.ecp.shape == (4, 3)


@pytest.mark.parametrize("num_points", [101, 201])
def test_check_tarp_atc_is_an_area_independent_of_the_grid(num_points):
    # ECP runs 0.1 above the diagonal until it saturates at 1. Over alpha in
    # [0.5, 1] the area between the curves is 0.1 * 0.4 + 0.1**2 / 2 = 0.045.
    alpha = jnp.linspace(0.0, 1.0, num_points)
    ecp = jnp.clip(alpha + 0.1, 0.0, 1.0)
    result = TARPResult(ecp=ecp, alpha=alpha, ecp_mean=ecp)

    atc, _ = check_tarp(result)

    assert atc == pytest.approx(0.045, abs=1e-3)


def _conjugate_gaussian_toy(scale, seed=0, num_sims=500, num_post=200, dim=2):
    """theta ~ N(0, 1), x = theta + N(0, 1): the posterior is N(x / 2, 1 / 2).

    ``scale`` multiplies the posterior standard deviation (1 is calibrated).
    """
    rng = np.random.default_rng(seed)
    thetas = rng.normal(size=(num_sims, dim))
    xs = thetas + rng.normal(size=(num_sims, dim))
    samples = xs / 2 + scale * np.sqrt(0.5) * rng.normal(size=(num_post, num_sims, dim))
    return thetas.astype(np.float32), samples.astype(np.float32)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_run_tarp_keeps_the_coverage_value_of_every_pair(bootstrap):
    thetas, posterior_samples = get_tarp_data()

    result = run_tarp(thetas, posterior_samples, bootstrap=bootstrap, num_bootstrap=4)

    coverage = np.asarray(result.coverage_values)
    assert coverage.shape == (thetas.shape[0],)
    assert result.num_posterior_samples == posterior_samples.shape[0]
    # each value is a count of posterior samples divided by their number
    counts = coverage * posterior_samples.shape[0]
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-4)
    assert coverage.min() >= 0.0 and coverage.max() <= 1.0


def test_check_tarp_ks_accepts_a_calibrated_posterior():
    thetas, samples = _conjugate_gaussian_toy(scale=1.0)
    _, ks_pval = check_tarp(run_tarp(thetas, samples))
    assert ks_pval > 1e-2


@pytest.mark.parametrize("scale", [0.7, 1.5])
def test_check_tarp_ks_rejects_a_miscalibrated_posterior(scale):
    thetas, samples = _conjugate_gaussian_toy(scale=scale)
    _, ks_pval = check_tarp(run_tarp(thetas, samples))
    assert ks_pval < 1e-3


def _curve_segments(ax, color):
    """x/y data of the lines of one colour on an axis, keyed by line style."""
    from matplotlib.colors import to_rgba

    return {
        line.get_linestyle(): line.get_data()
        for line in ax.get_lines()
        if to_rgba(line.get_color()) == to_rgba(color)
    }


def test_confidence_panel_reports_the_measured_coverage_at_three_sigma():
    """The 3 sigma reading must come from the pairs, not from interpolating a 30-bin curve."""
    from scipy.stats import norm

    thetas, samples = _conjugate_gaussian_toy(scale=0.7, num_sims=3000, num_post=500)
    result = run_tarp(thetas, samples)

    fig, ax = plot_tarp(result, mode="confidence")
    segments = _curve_segments(ax, "#202A44")
    z = np.concatenate([seg[0] for seg in segments.values()])
    z_hat = np.concatenate([seg[1] for seg in segments.values()])
    order = np.argsort(z)
    plotted = 2 * norm.cdf(np.interp(3.0, z[order], z_hat[order])) - 1
    plt.close(fig)

    # A pair whose truth is closer to the reference than r of the N samples is
    # spread over ((r, r + 1) / (N + 1)), so it is covered at level c with
    # probability clip(c * (N + 1) - r, 0, 1).
    num_post = samples.shape[0]
    counts = np.round(np.asarray(result.coverage_values, dtype=np.float64) * num_post)
    measured = np.mean(np.clip(0.9973 * (num_post + 1) - counts, 0.0, 1.0))
    assert measured < 0.985  # clearly overconfident at 3 sigma (nominal 0.9973)
    assert abs(plotted - measured) < 0.004


def test_confidence_panel_is_dashed_beyond_the_posterior_sample_resolution():
    """With N posterior samples, tail masses below 1 / (N + 1) are extrapolated."""
    thetas, samples = _conjugate_gaussian_toy(scale=1.0, num_sims=400, num_post=100)
    result = run_tarp(thetas, samples)

    fig, ax = plot_tarp(result, mode="confidence")
    segments = _curve_segments(ax, "#202A44")
    plt.close(fig)

    z_resolved = 2.58  # two-sided z of a tail mass of 1 / 101
    assert segments["-"][0].max() <= z_resolved
    assert segments["--"][0].min() >= z_resolved - 0.1  # dashed part continues the solid one
    assert segments["--"][0].max() > 3.0
