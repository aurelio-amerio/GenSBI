import matplotlib.pyplot as plt
import numpy as np
import pytest

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
