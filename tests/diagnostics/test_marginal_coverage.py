import pytest
import numpy as np
import matplotlib.pyplot as plt
from gensbi.diagnostics.marginal_coverage import (
    compute_marginal_coverage,
    plot_marginal_coverage,
)


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


def test_plot_marginal_coverage():
    """Test that plotting runs without error."""
    D_dim = 3
    N_batch = 50
    alpha = np.random.rand(D_dim, N_batch)

    fig = plot_marginal_coverage(alpha)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# Coverage improvement tests


def test_kde_truth_out_of_range():
    """Truth outside posterior grid should give alpha=1 (or close)."""
    N_batch = 5
    D_dim = 1
    N_samples = 200

    # Posterior concentrated around 0, truth at 100 (way outside)
    posterior_samples = np.random.randn(N_samples, N_batch, D_dim)
    theta = np.ones((N_batch, D_dim)) * 100.0

    alpha = compute_marginal_coverage(theta, posterior_samples, method="KDE")
    assert alpha.shape == (D_dim, N_batch)
    # alpha should be very close to 1 since truth is far outside
    assert np.all(alpha >= 0.9)


def test_histogram_truth_out_of_range():
    """Truth outside histogram bins should give alpha=1."""
    N_batch = 5
    D_dim = 1
    N_samples = 200

    # Posterior concentrated around 0, truth at 100
    posterior_samples = np.random.randn(N_samples, N_batch, D_dim)
    theta = np.ones((N_batch, D_dim)) * 100.0

    alpha = compute_marginal_coverage(theta, posterior_samples, method="histogram")
    assert alpha.shape == (D_dim, N_batch)
    assert np.all(alpha == 1.0)


def test_kde_exception_handling():
    """Degenerate samples (all identical) should trigger KDE exception path."""
    N_batch = 2
    D_dim = 1
    N_samples = 50

    # All identical samples → KDE can't fit properly
    posterior_samples = np.ones((N_samples, N_batch, D_dim))
    theta = np.zeros((N_batch, D_dim))

    alpha = compute_marginal_coverage(theta, posterior_samples, method="KDE")
    assert alpha.shape == (D_dim, N_batch)
    # Should either be NaN (from exception) or some value — no crash


def test_plot_marginal_coverage_axes_hiding():
    """Test with D_dim not divisible by n_cols to trigger axis hiding."""
    D_dim = 2  # 2 dims, n_cols=3 → 1 row, 3 columns, 1 hidden
    N_batch = 30
    alpha = np.random.rand(D_dim, N_batch)

    fig = plot_marginal_coverage(alpha, n_cols=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

