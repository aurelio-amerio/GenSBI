import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

from gensbi.diagnostics.marginal_coverage import (
    compute_marginal_coverage,
    estimate_hat_z,
    plot_marginal_coverage,
)

METHODS = ["histogram", "KDE"]


def _draw(shape, rng, size):
    """Draw from one of the toy posterior shapes used in the calibration tests."""
    if shape == "gaussian":
        return rng.normal(size=size)
    if shape == "uniform":
        return rng.uniform(size=size)
    if shape == "bimodal":
        return 0.5 * rng.normal(size=size) + rng.choice([-2.0, 2.0], size=size)
    raise ValueError(shape)


def test_compute_marginal_coverage_shape():
    """Test that the output shape of compute_marginal_coverage is correct."""
    N_batch = 10
    D_dim = 2
    N_samples = 100

    theta = np.zeros((N_batch, D_dim))
    posterior_samples = np.random.randn(N_samples, N_batch, D_dim)

    # Test default (histogram)
    alpha = compute_marginal_coverage(theta, posterior_samples)
    assert alpha.shape == (D_dim, N_batch)
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= 1.0)

    # Test explicit histogram method
    alpha_hist = compute_marginal_coverage(theta, posterior_samples, method="histogram")
    assert alpha_hist.shape == (D_dim, N_batch)
    assert np.all(alpha_hist >= 0.0)
    assert np.all(alpha_hist <= 1.0)

    # Test KDE method
    alpha_kde = compute_marginal_coverage(theta, posterior_samples, method="KDE")
    assert alpha_kde.shape == (D_dim, N_batch)
    assert np.all(alpha_kde >= 0.0)
    assert np.all(alpha_kde <= 1.0)

    # Test invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        compute_marginal_coverage(theta, posterior_samples, method="invalid")


# Calibration: the credibility must be Uniform(0, 1) for an exact posterior


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", ["gaussian", "uniform", "bimodal"])
def test_calibrated_posterior_gives_uniform_credibility(method, shape):
    """Truth and samples from the same distribution must give alpha ~ U(0, 1).

    Few posterior samples on purpose: this is where estimators that rank the
    truth against densities built from the samples alone read as overconfident.
    """
    rng = np.random.default_rng(0)
    n_obs, n_samples = 600, 50
    theta = _draw(shape, rng, (n_obs, 1))
    samples = _draw(shape, rng, (n_samples, n_obs, 1))

    alpha = compute_marginal_coverage(theta, samples, method=method)[0]

    assert np.all(np.isfinite(alpha))
    assert kstest(alpha, "uniform").pvalue > 1e-3
    for level in (0.6827, 0.9545):
        sigma = np.sqrt(level * (1 - level) / n_obs)
        assert abs((alpha <= level).mean() - level) < 4 * sigma


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "scale, exact_coverage",
    [
        (0.5, 0.3829),  # too narrow: 2 * Phi(0.5) - 1
        (2.0, 0.9545),  # too wide:   2 * Phi(2.0) - 1
    ],
)
def test_miscalibrated_posterior_is_detected(method, scale, exact_coverage):
    """The 68.27% HPD region of N(0, scale^2) must cover N(0, 1) truths at the exact rate."""
    rng = np.random.default_rng(1)
    n_obs, n_samples = 400, 200
    theta = rng.normal(size=(n_obs, 1))
    samples = scale * rng.normal(size=(n_samples, n_obs, 1))

    alpha = compute_marginal_coverage(theta, samples, method=method)[0]

    assert abs((alpha <= 0.6827).mean() - exact_coverage) < 0.08


@pytest.mark.parametrize("method", METHODS)
def test_credibility_is_strictly_inside_unit_interval(method):
    """Randomised ranks never return exactly 0 or 1."""
    rng = np.random.default_rng(2)
    theta = rng.normal(size=(50, 1))
    samples = rng.normal(size=(100, 50, 1))

    alpha = compute_marginal_coverage(theta, samples, method=method)

    assert np.all(alpha > 0.0)
    assert np.all(alpha < 1.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("truth", [100.0, 1e9])
def test_truth_far_outside_samples_gives_credibility_near_one(method, truth):
    """A truth the posterior assigns no mass to is the clearest overconfidence signal."""
    rng = np.random.default_rng(3)
    n_samples, n_obs = 200, 5
    samples = rng.normal(size=(n_samples, n_obs, 1))
    theta = np.full((n_obs, 1), truth)

    alpha = compute_marginal_coverage(theta, samples, method=method)

    # every sample is denser than the truth: alpha >= N / (N + 1)
    assert np.all(alpha > 0.99)


@pytest.mark.parametrize("method", METHODS)
def test_collapsed_posterior_with_truth_elsewhere_gives_credibility_near_one(method):
    """All-identical samples must not turn a missed truth into a NaN."""
    samples = np.ones((50, 3, 1))
    theta = np.zeros((3, 1))

    alpha = compute_marginal_coverage(theta, samples, method=method)

    assert np.all(alpha > 0.9)


def test_histogram_fixed_bin_edges_truth_outside_edges():
    """With user-supplied edges, a truth outside them ranks below every sample."""
    rng = np.random.default_rng(4)
    samples = rng.uniform(-1, 1, size=(200, 4, 1))
    theta = np.full((4, 1), 5.0)

    alpha = compute_marginal_coverage(
        theta, samples, method="histogram", bins=np.linspace(-1, 1, 9)
    )

    assert np.all(alpha > 0.99)


# Reproducibility of the random tie-breaking


@pytest.mark.parametrize("method", METHODS)
def test_seed_controls_tie_breaking(method):
    rng = np.random.default_rng(5)
    # a flat posterior is tie-heavy, so the result depends strongly on the draw
    theta = rng.uniform(size=(40, 2))
    samples = rng.uniform(size=(100, 40, 2))

    a0 = compute_marginal_coverage(theta, samples, method=method, seed=0)
    a0_again = compute_marginal_coverage(theta, samples, method=method, seed=0)
    a1 = compute_marginal_coverage(theta, samples, method=method, seed=1)

    np.testing.assert_array_equal(a0, a0_again)
    assert not np.array_equal(a0, a1)


# Input handling


@pytest.mark.parametrize("method", METHODS)
def test_wrong_shapes_raise(method):
    theta = np.zeros((10, 2))
    with pytest.raises(ValueError, match="shape"):
        compute_marginal_coverage(theta, np.zeros((100, 9, 2)), method=method)
    with pytest.raises(ValueError, match="shape"):
        compute_marginal_coverage(theta, np.zeros((100, 10, 3)), method=method)


def test_invalid_estimator_arguments_raise():
    """A bad binning rule or bandwidth is a caller error, not a NaN result."""
    rng = np.random.default_rng(8)
    theta = rng.normal(size=(3, 1))
    samples = rng.normal(size=(100, 3, 1))

    with pytest.raises(ValueError):
        compute_marginal_coverage(theta, samples, method="histogram", bins="not-a-rule")
    with pytest.raises(ValueError):
        compute_marginal_coverage(theta, samples, method="KDE", bw="not-a-rule")


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_values_give_nan_and_warn(method):
    rng = np.random.default_rng(6)
    theta = rng.normal(size=(4, 1))
    samples = rng.normal(size=(100, 4, 1))
    samples[0, 1, 0] = np.nan
    theta[3, 0] = np.inf

    with pytest.warns(UserWarning, match="could not be computed"):
        alpha = compute_marginal_coverage(theta, samples, method=method)[0]

    assert np.isnan(alpha[1]) and np.isnan(alpha[3])
    assert np.all(np.isfinite(alpha[[0, 2]]))


@pytest.mark.parametrize("method", METHODS)
def test_show_progress_bar_false_is_silent(method, capsys):
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(3, 2))
    samples = rng.normal(size=(100, 3, 2))

    compute_marginal_coverage(theta, samples, method=method, show_progress_bar=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_estimate_hat_z_ignores_nan_credibilities():
    """NaN credibilities are excluded, not counted as misses."""
    alpha = np.array([0.1, 0.9, np.nan])

    with pytest.warns(UserWarning, match="NaN"):
        hat_z = estimate_hat_z(alpha, nbins=2, zmax=1)

    # at z = 1 (68.27%): 1 of the 2 finite values is covered -> 50% -> z = 0.6745
    np.testing.assert_allclose(hat_z["mean"][1], 0.6745, atol=1e-4)


# Plotting


def test_plot_marginal_coverage():
    """Test that plotting runs without error."""
    D_dim = 3
    N_batch = 50
    alpha = np.random.rand(D_dim, N_batch)

    fig = plot_marginal_coverage(alpha)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_marginal_coverage_axes_hiding():
    """Test with D_dim not divisible by n_cols to trigger axis hiding."""
    D_dim = 2  # 2 dims, n_cols=3 → 1 row, 3 columns, 1 hidden
    N_batch = 30
    alpha = np.random.rand(D_dim, N_batch)

    fig = plot_marginal_coverage(alpha, n_cols=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_marginal_coverage_titles_use_parameter_labels():
    alpha = np.random.default_rng(0).uniform(size=(2, 30))

    fig = plot_marginal_coverage(alpha, parameter_labels=["As", "n1"])
    assert [ax.get_title() for ax in fig.axes[:2]] == ["As", "n1"]
    plt.close(fig)

    fig = plot_marginal_coverage(alpha)
    assert [ax.get_title() for ax in fig.axes[:2]] == ["Dim 1", "Dim 2"]
    plt.close(fig)


def test_plot_marginal_coverage_rejects_wrong_number_of_labels():
    alpha = np.random.default_rng(0).uniform(size=(2, 30))
    with pytest.raises(ValueError, match="parameter_labels"):
        plot_marginal_coverage(alpha, parameter_labels=["As"])


def _black_curve_styles(ax):
    from matplotlib.colors import to_rgba

    return {
        line.get_linestyle(): line.get_data()
        for line in ax.get_lines()
        if to_rgba(line.get_color()) == to_rgba("k")
    }


def test_plot_marginal_coverage_is_dashed_beyond_the_posterior_sample_resolution():
    """With N posterior samples, tail masses below 1 / (N + 1) are extrapolated."""
    alpha = np.random.default_rng(0).uniform(size=(1, 400))

    fig = plot_marginal_coverage(alpha, num_posterior_samples=100)
    segments = _black_curve_styles(fig.axes[0])
    plt.close(fig)

    z_resolved = 2.58  # two-sided z of a tail mass of 1 / 101
    assert segments["-"][0].max() <= z_resolved
    assert segments["--"][0].min() >= z_resolved - 0.1
    assert segments["--"][0].max() > 3.0


def test_plot_marginal_coverage_is_solid_without_num_posterior_samples():
    alpha = np.random.default_rng(0).uniform(size=(1, 400))

    fig = plot_marginal_coverage(alpha)
    segments = _black_curve_styles(fig.axes[0])
    plt.close(fig)

    assert set(segments) == {"-"}
