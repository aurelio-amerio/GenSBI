import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from tqdm import tqdm

from gensbi.diagnostics.utils import (
    alpha_from_z,
    jefferys_interval,
    probit,
    z_from_alpha,
)


def _compute_marginal_coverage_KDE(
    theta, posterior_samples, grid_points=2048, bw="ISJ"
):
    """
    Computes the marginal coverage (credibility) for each observation and dimension
    using Kernel Density Estimation.

    For each observation b and dimension d, estimates the 1D marginal posterior
    density p(x) via KDE, then computes the HPD (Highest Posterior Density)
    credibility:

    .. math::
        \alpha = \int_{p(x) > p(\theta^*)} p(x) dx

    This gives the probability mass in regions denser than the true value.
    If the posterior is well calibrated, alpha should be Uniform(0, 1).

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    grid_points : int, optional
        Number of grid points for KDE evaluation. Default is 2048.
    bw : str or float, optional
        Bandwidth method or value for KDE. Default is "ISJ"
        (Improved Sheather-Jones, a plug-in bandwidth selector).

    Returns
    -------
    alpha : np.ndarray of shape (D_dim, N_batch)
        The credibility values (0 to 1).
        If the posterior is well calibrated, these should be uniformly distributed.
    """

    theta = np.asarray(theta)
    posterior_samples = np.asarray(posterior_samples)

    N_samples, N_batch, D_dim = posterior_samples.shape

    assert theta.shape[0] == N_batch
    assert theta.shape[1] == D_dim

    alpha_values = np.zeros((D_dim, N_batch))

    estimator = FFTKDE(bw=bw)

    for d in range(D_dim):
        print(f"Computing coverage for dimension {d+1}/{D_dim}")
        for b in tqdm(range(N_batch), leave=False):
            samples = posterior_samples[:, b, d]
            truth = theta[b, d]

            try:
                kde_fit = estimator.fit(samples)
                x_grid, y_grid = kde_fit.evaluate(grid_points)

                # Evaluate p(θ*) by linear interpolation on the KDE grid.
                # If truth falls outside the grid, the posterior assigns
                # negligible density to it, so alpha → 1.
                if truth < x_grid.min() or truth > x_grid.max():
                    pdf_truth = 0.0
                else:
                    pdf_truth = np.interp(truth, x_grid, y_grid)

                # Approximate ∫_{p(x) ≥ p(θ*)} p(x) dx via a Riemann sum
                # on the uniform KDE grid. The grid spacing cancels in the
                # ratio, so we can use the raw density values directly.
                mask = y_grid >= pdf_truth
                alpha = y_grid[mask].sum() / y_grid.sum()

                alpha_values[d, b] = alpha

            except Exception as e:
                print(f"Error in KDE for batch {b}, dim {d}: {e}")
                alpha_values[d, b] = np.nan

    return alpha_values


def _compute_marginal_coverage_histogram(theta, posterior_samples, bins="stone"):
    """
    Computes marginal coverage using empirical histograms.

    This is a discrete approximation of the HPD credibility: for each
    (observation, dimension) pair, it bins the posterior samples and
    sums the mass of all bins at least as dense as the one containing
    the true parameter. Faster than KDE and works well for multimodal
    distributions, but accuracy depends on bin count.

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    bins : int or str, optional
        Number of bins or binning strategy. See ``numpy.histogram_bin_edges``
        for valid string values ('stone', 'scott', 'fd', 'sturges', etc).
        Default is 'stone'.

    Returns
    -------
    alpha_values : np.ndarray of shape (D_dim, N_batch)
        The credibility values (0 to 1).
    """
    theta = np.asarray(theta)
    posterior_samples = np.asarray(posterior_samples)
    N_samples, N_batch, D_dim = posterior_samples.shape

    alpha_values = np.zeros((D_dim, N_batch))

    for d in range(D_dim):
        print(f"Computing coverage for dimension {d+1}/{D_dim}")
        for b in tqdm(range(N_batch), leave=False):
            samples = posterior_samples[:, b, d]
            truth = theta[b, d]

            counts, bin_edges = np.histogram(samples, bins=bins)

            # Locate the bin containing the true parameter.
            # np.digitize returns 1-based indices, so subtract 1.
            bin_idx = np.digitize(truth, bin_edges) - 1

            if bin_idx < 0 or bin_idx >= len(counts):
                # Truth lies outside the histogram range, meaning the
                # posterior assigns zero density there → alpha = 1.
                alpha = 1.0
            else:
                truth_count = counts[bin_idx]
                # HPD credibility: sum all bins with count ≥ truth bin count,
                # normalized by total samples. This is a discrete analogue of
                # ∫_{p(x) ≥ p(θ*)} p(x) dx.
                mask = counts >= truth_count
                alpha = counts[mask].sum() / N_samples

            alpha_values[d, b] = alpha

    return alpha_values


def compute_marginal_coverage(theta, posterior_samples, method="histogram", **kwargs):
    """
    Compute marginal coverage for each observation and dimension.

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    method : str, optional
        Method to use for computing marginal coverage. Options are:
        - "histogram": Use empirical histograms (default).
        - "KDE": Use Kernel Density Estimation.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the chosen method.

    Returns
    -------
    alpha_values : np.ndarray of shape (D_dim, N_batch)
        The credibility values (0 to 1).
    """
    if method == "histogram":
        return _compute_marginal_coverage_histogram(theta, posterior_samples, **kwargs)
    elif method == "KDE":
        return _compute_marginal_coverage_KDE(theta, posterior_samples, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_hat_z(alpha_values, nbins=50, zmax=4, z_band=1):
    """
    Compute empirical coverage (hat{z}) vs nominal coverage (z) for one dimension.

    Given a set of HPD credibility values (alpha), this function builds the
    empirical CDF of those values and converts it to z-score space. For a
    well-calibrated posterior, hat{z} should equal z along the diagonal.

    Parameters
    ----------
    alpha_values : array_like
        1D array of credibility values for a single dimension.
    nbins : int, optional
        Number of bins for z (default is 50).
    zmax : float, optional
        Maximum z value to evaluate (default is 4).
    z_band : float, optional
        Z-score for the Jeffrey's uncertainty band (default is 1, ~68% CI).

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'z': Nominal coverage z-values.
        - 'mean': Empirical coverage hat{z}.
        - 'upper': Upper bound of uncertainty band.
        - 'lower': Lower bound of uncertainty band.
    """
    n = len(alpha_values)
    zlist = np.linspace(0, zmax, nbins)

    # Map z-scores to coverage thresholds: z → alpha → coverage = 1 - alpha.
    # At each threshold t, count how many observations have alpha < t
    # (i.e., truth lies within the t-credibility region).
    tlist = 1 - alpha_from_z(zlist)
    k = np.array([np.sum(alpha_values < t) for t in tlist])

    # Empirical coverage fraction and its Jeffrey's uncertainty band
    r_mean = k / n
    r_lower, r_upper = jefferys_interval(k, n, z=z_band)

    # Convert coverage fractions back to z-scores for plotting
    z_mean = z_from_alpha(1 - r_mean)
    z_band_lower = z_from_alpha(1 - r_lower)
    z_band_upper = z_from_alpha(1 - r_upper)

    return dict(z=zlist, mean=z_mean, upper=z_band_upper, lower=z_band_lower)


def plot_marginal_coverage(alpha_values, zmax=3.5, n_cols=3, figsize=None):
    """
    Plots the marginal coverage for all dimensions.

    Parameters
    ----------
    alpha_values : np.ndarray of shape (D_dim, N_batch)
        Credibility values.
    zmax : float, optional
        Maximum z-score for plotting limits (default is 3.5).
    n_cols : int, optional
        Number of columns in the subplot grid (default is 3).
    figsize : tuple, optional
        Figure size (width, height). If None, calculated automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """

    n_cols = min(n_cols, alpha_values.shape[0])

    D_dim, N_batch = alpha_values.shape

    n_rows = (D_dim + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)  # handle 1D case

    axes_flat = axes.flatten()

    for k in range(D_dim, len(axes_flat)):
        axes_flat[k].axis("off")

    for d in range(D_dim):
        ax = axes_flat[d]

        hat_z = estimate_hat_z(alpha_values[d], zmax=zmax)
        z_grid = hat_z["z"]

        ax.plot(z_grid, hat_z["mean"], "k")

        upper = hat_z["upper"]
        upper = np.where(np.isinf(upper), 100.0, upper)  # clamp infs for plotting
        ax.fill_between(z_grid, hat_z["lower"], upper, color="0.8")

        ax.plot([0, zmax + 0.5], [0, zmax + 0.5], "--", color="darkgreen")

        # Annotate integer sigma levels with empirical coverage percentages
        for t in range(1, int(zmax) + 1):
            l = np.interp(t, z_grid, hat_z["mean"])

            if not np.isinf(l):
                ax.plot([0, t], [l, l], ":", color="r")
                c = 1 - alpha_from_z(l)
                ax.text(
                    0.1,
                    l + 0.02,
                    ("%.2f" % (c * 100)) + "%",
                    ha="left",
                    va="bottom",
                )
                ax.plot([t, t], [0, l], ":", color="r")
            else:
                ax.plot([t, t], [0, 10.0], ":", color="r")

            c = 1 - alpha_from_z(t)
            ax.text(
                t + 0.02,
                0.1,
                ("%.2f" % (c * 100)) + "%",
                rotation=-90,
                ha="left",
                va="bottom",
            )

        ax.set_xlim([0, zmax])
        ax.set_ylim([0, zmax + 0.5])
        ax.set_ylabel(r"Empirical coverage, $\hat{z}$")
        ax.set_xlabel(r"Confidence level, $z$")
        ax.set_title(f"Dim {d+1}")

        # Compute rotation angle for diagonal labels
        x_ = [0, 10]
        y_ = [0, 10]
        p1 = ax.transData.transform((x_[0], y_[0]))
        p2 = ax.transData.transform((x_[1], y_[1]))
        dy_screen = p2[1] - p1[1]
        dx_screen = p2[0] - p1[0]
        phi = np.degrees(np.arctan2(dy_screen, dx_screen))

        ax.text(
            zmax / 2,
            zmax / 2 + 0.4,
            "Conservative",
            ha="center",
            va="center",
            rotation=phi,
            color="darkgreen",
            alpha=1,
            rotation_mode="anchor",
            fontsize=11,
        )
        ax.text(
            zmax / 2 + 0.4,
            zmax / 2,
            "Overconfident",
            ha="center",
            va="center",
            rotation=phi,
            color="darkgreen",
            alpha=1,
            rotation_mode="anchor",
            fontsize=11,
        )

    plt.tight_layout()
    return fig
