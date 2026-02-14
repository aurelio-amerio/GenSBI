import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from tqdm import tqdm


def _compute_marginal_coverage_KDE(
    theta, posterior_samples, grid_points=2048, bw="ISJ"
):
    """
    Computes the marginal coverage (credibility) for each observation and dimension.

    For each observation b and dimension d, it estimates the posterior density p(x)
    using KDE, and computes the probability mass of the region where p(x) > p(theta_bd).

    .. math::
        \alpha = \int_{p(x) > p(\theta^*)} p(x) dx

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    grid_points : int, optional
        Number of grid points for KDE evaluation. Default is 2048.
    bw : str or float, optional
        Bandwidth method or value for KDE. Default is "ISJ".

    Returns
    -------
    alpha : np.ndarray of shape (D_dim, N_batch)
        The credibility values (0 to 1).
        If the posterior is well calibrated, these should be uniformly distributed.
    """
    # specific imports

    theta = np.asarray(theta)
    posterior_samples = np.asarray(posterior_samples)

    N_samples, N_batch, D_dim = posterior_samples.shape

    assert theta.shape[0] == N_batch
    assert theta.shape[1] == D_dim

    alpha_values = np.zeros((D_dim, N_batch))

    # We iterate over dimensions and batches.
    # This might be slow for very large batches, but KDE is the bottleneck.
    # We can use tqdm to show progress.

    estimator = FFTKDE(bw=bw)

    for d in range(D_dim):
        print(f"Computing coverage for dimension {d+1}/{D_dim}")
        for b in tqdm(range(N_batch), leave=False):
            samples = posterior_samples[:, b, d]
            truth = theta[b, d]

            # Fit KDE
            # FFTKDE is fast.
            # We treat samples as 1D data.
            try:
                # grid points: auto or fixed?
                # The original code used 5000.
                # We can let KDEpy decide or enforce a grid.
                # evaluate() returns x, y
                kde_fit = estimator.fit(samples)
                x_grid, y_grid = kde_fit.evaluate(grid_points)

                # We need to evaluate density at the true value 'truth'.
                # We can interpolate y_grid at 'truth'.
                # We assume sorted x_grid from FFTKDE.

                # Check bounds
                if truth < x_grid.min() or truth > x_grid.max():
                    # If truth is outside support of KDE (numerically), density is ~0.
                    # Then the integral of p(x) > 0 is 1. (Coverage is 100% or 0%?)
                    # If p(theta) is 0, then region p(x) > 0 contains everything -> alpha = 1.
                    # But usually this means our posterior missed the truth completely.
                    # If we follow the "highest predictive density" logic:
                    # The value p(theta) is small.
                    # The area where p(x) > p(theta) is large (almost 1).
                    # So alpha should be close to 1.
                    pdf_truth = 0.0
                else:
                    # Linear interpolation for pdf value at truth
                    # We can use numpy interp
                    pdf_truth = np.interp(truth, x_grid, y_grid)

                # Calculate integral \int_{p(x) > p(theta)} p(x) dx
                # approximated by sum of y_grid where y_grid > pdf_truth, weighted by dx.
                # or simply sum(y_grid[mask]) / sum(y_grid).

                mask = y_grid >= pdf_truth
                alpha = y_grid[mask].sum() / y_grid.sum()

                alpha_values[d, b] = alpha

            except Exception as e:
                print(f"Error in KDE for batch {b}, dim {d}: {e}")
                alpha_values[d, b] = np.nan

    return alpha_values


def _compute_marginal_coverage_histogram(theta, posterior_samples, bins="stone"):
    """
    Computes marginal coverage for multimodal distributions using empirical histograms.

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    bins : int or str, optional
        Number of bins or method name (e.g. 'stone', 'scott', 'fd', 'sturges', etc) for histogram binning. See [numpy.histogram_bin_edges](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html) for details.
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

            # 1. Create histogram (counts represent unnormalized density)
            counts, bin_edges = np.histogram(samples, bins=bins)

            # 2. Find which bin the truth falls into
            # digitize returns 1-based indices, so we subtract 1
            bin_idx = np.digitize(truth, bin_edges) - 1

            # 3. Handle boundary conditions
            if bin_idx < 0 or bin_idx >= len(counts):
                # Truth is completely outside the sampled posterior range.
                # Its density is effectively 0, so the entire posterior mass
                # has a higher density than the truth.
                alpha = 1.0
            else:
                # 4. Find the 'density' (count) at the truth
                truth_count = counts[bin_idx]

                # 5. Sum the mass of all regions denser than the truth
                # Equivalent to \int_{p(x) > p(\theta)} p(x) dx
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


def probit(x):
    """
    Compute the probit function (inverse CDF of standard normal).

    Parameters
    ----------
    x : array_like
        Input values in (0, 1).

    Returns
    -------
    y : array_like
        The probit values.
    """
    return scipy.stats.norm.ppf(x)


def z_from_alpha(alpha):
    """
    Convert alpha (credibility) to z-score (standard deviations).

    Parameters
    ----------
    alpha : array_like
        Credibility values.

    Returns
    -------
    z : array_like
        Corresponding z-scores.
    """
    return probit(1 - alpha / 2)


def alpha_from_z(z):
    """
    Convert z-score to alpha (credibility).

    Parameters
    ----------
    z : array_like
        Z-scores.

    Returns
    -------
    alpha : array_like
        Corresponding credibility values.
    """
    return 2 - scipy.stats.norm.cdf(z) * 2


def jefferys_interval(k, n, z=1):
    """
    Compute Jefferys interval for a binomial proportion.

    Parameters
    ----------
    k : array_like
        Number of successes.
    n : int or array_like
        Total number of trials.
    z : float, optional
        Z-score for the interval (default is 1).

    Returns
    -------
    interval : np.ndarray of shape (..., 2)
        Lower and upper bounds of the interval.
    """
    alpha = alpha_from_z(z=z)
    lower = scipy.stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = scipy.stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return np.array([np.where(k > 0, lower, 0.0), np.where(k < n, upper, 1.0)]).T


def estimate_hat_z(alpha_values, nbins=50, zmax=4, z_band=1):
    """
    Compute the empirical coverage (hat{z}) vs nominal coverage (z).

    Parameters
    ----------
    alpha_values : array_like
        1D array of credibility values for a single dimension.
    nbins : int, optional
        Number of bins for z (default is 50).
    zmax : float, optional
        Maximum z value to evaluate (default is 4).
    z_band : float, optional
        Z-score for the uncertainty band (default is 1).

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
    tlist = 1 - alpha_from_z(zlist)

    # alpha_values are 'alpha' (0 is best, 1 is worst? No.)
    # In original code:
    # gamma.append(yy[yy >= f(z[i][j])].sum() / yy.sum()) -> this is "alpha" or "1-alpha"?
    # If p(theta) is max, term inside sum is small?
    # Wait.
    # If p(theta) is peak, then y_grid >= p(theta) is only the peak.
    # So sum is small -> alpha close to 0. (Ideal)
    # If p(theta) is tail, then y_grid >= p(theta) is almost everything.
    # So sum is large -> alpha close to 1.
    # So 'alpha' behaves like a p-value. Uniform(0,1).

    # Original code:
    # k = np.array([sum(masses < t) for t in tlist])
    # tlist goes from small (high z) to large (low z).
    # tlist = 1 - alpha_from_z(zlist)
    # z=0 -> alpha_from_z=1 -> tlist=0
    # z=large -> alpha_from_z=0 -> tlist=1

    # So we count how many alpha_values are smaller than t.
    # This is calculating the CDF of alpha_values.

    k = np.array([np.sum(alpha_values < t) for t in tlist])

    r_mean = k / n
    r_band = jefferys_interval(k, n, z=z_band)

    z_mean = z_from_alpha(1 - r_mean)

    # 1 - r_band is (N, 2)
    # z_from_alpha expects scalar or array.
    # We need to broadcast carefully.
    # z_from_alpha(p) = probit(1 - p/2).
    # If p is close to 1 (r small), z is small.
    # If p is close to 0 (r large), z is large.

    # careful with broadcasting if z_from_alpha usage
    z_band_lower = z_from_alpha(1 - r_band[:, 0])
    z_band_upper = z_from_alpha(1 - r_band[:, 1])

    # Note: original code returned z_band[:, 1] as upper.
    # The interval function returns (lower, upper) probabilities.
    # We convert probabilities to Z-scores.

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

    # Hide unused axes
    for k in range(D_dim, len(axes_flat)):
        axes_flat[k].axis("off")

    for d in range(D_dim):
        ax = axes_flat[d]

        # Calculate stats
        hat_z = estimate_hat_z(alpha_values[d], zmax=zmax)
        z_grid = hat_z["z"]

        # Plot mean
        ax.plot(z_grid, hat_z["mean"], "k")

        # Plot band
        upper = hat_z["upper"]
        # Fix infs
        upper = np.where(np.isinf(upper), 100.0, upper)
        ax.fill_between(z_grid, hat_z["lower"], upper, color="0.8")

        # Diagonal reference
        ax.plot([0, zmax + 0.5], [0, zmax + 0.5], "--", color="darkgreen")

        # Mark integers
        for t in range(1, int(zmax) + 1):
            # Interpolate mean at integer t
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

        # compute the rotation angle
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
