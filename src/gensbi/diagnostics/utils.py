"""Shared statistical utilities for diagnostic functions."""

import warnings

import numpy as np
import scipy.stats


def probit(x):
    """
    Inverse CDF of the standard normal distribution.

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
    Convert significance level alpha to z-score.

    Parameters
    ----------
    alpha : array_like
        Significance levels.

    Returns
    -------
    z : array_like
        Corresponding z-scores.
    """
    return probit(1 - alpha / 2)


def alpha_from_z(z):
    """
    Convert z-score to significance level alpha.

    Parameters
    ----------
    z : array_like
        Z-scores.

    Returns
    -------
    alpha : array_like
        Corresponding significance levels.
    """
    return 2 - scipy.stats.norm.cdf(z) * 2


def jefferys_interval(k, n, z=1):
    """
    Compute Jeffrey's interval for a binomial proportion.

    Uses the Beta distribution with Jeffrey's prior (Beta(0.5, 0.5)).

    Parameters
    ----------
    k : array_like
        Number of successes.
    n : int or array_like
        Total number of trials.
    z : float, optional
        Z-score controlling the interval width (default is 1, i.e. ~68% CI).

    Returns
    -------
    lower : np.ndarray
        Lower bounds of the interval.
    upper : np.ndarray
        Upper bounds of the interval.
    """
    alpha = alpha_from_z(z=z)
    lower = scipy.stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = scipy.stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)

    lower = np.where(k > 0, lower, 0.0)
    upper = np.where(k < n, upper, 1.0)

    return lower, upper


def estimate_hat_z(alpha_values, nbins=50, zmax=4, z_band=1):
    """
    Compute empirical coverage (hat{z}) vs nominal coverage (z) for one dimension.

    Given a set of HPD credibility values (alpha), this function builds the
    empirical CDF of those values and converts it to z-score space. For a
    well-calibrated posterior, hat{z} should equal z along the diagonal.

    Parameters
    ----------
    alpha_values : array_like
        1D array of credibility values for a single dimension. NaN values
        are excluded, with a warning.
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
    alpha_values = np.asarray(alpha_values)
    is_nan = np.isnan(alpha_values)
    if is_nan.any():
        warnings.warn(
            f"{int(is_nan.sum())} of {alpha_values.size} credibility values are NaN "
            "and are excluded from the coverage estimate.",
            UserWarning,
            stacklevel=2,
        )
        alpha_values = alpha_values[~is_nan]

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


def resolution_limit_z(num_posterior_samples):
    """
    Largest confidence level, as a z-score, that the posterior samples resolve.

    A credibility computed from N posterior samples is a randomised rank with
    cells of width 1 / (N + 1). Coverage levels whose tail mass is below
    1 / (N + 1) fall inside the last cell: the empirical coverage there is an
    extrapolation that assumes the true values are spread evenly within it.

    Parameters
    ----------
    num_posterior_samples : int
        Number of posterior samples per observation.

    Returns
    -------
    z : float
        The two-sided z-score of a tail mass of ``1 / (num_posterior_samples + 1)``.
    """
    return float(z_from_alpha(1.0 / (num_posterior_samples + 1)))


def plot_coverage_curve(ax, z, z_hat, z_resolved=None, **kwargs):
    """
    Plot an empirical coverage curve, dashed where it is extrapolated.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    z : array_like
        Nominal confidence levels as z-scores, increasing.
    z_hat : array_like
        Empirical coverage as z-scores.
    z_resolved : float, optional
        Resolution limit from :func:`resolution_limit_z`. The curve is solid up
        to it and dashed beyond. If None, the whole curve is solid.
    **kwargs
        Passed to ``ax.plot`` (colour, label, ...).
    """
    z, z_hat = np.asarray(z), np.asarray(z_hat)
    if z_resolved is None:
        ax.plot(z, z_hat, linestyle="-", **kwargs)
        return

    num_solid = int(np.count_nonzero(z <= z_resolved))
    ax.plot(z[:num_solid], z_hat[:num_solid], linestyle="-", **kwargs)
    if num_solid < len(z):
        kwargs.pop("label", None)
        # start at the last solid point so that the two parts join
        start = max(num_solid - 1, 0)
        ax.plot(z[start:], z_hat[start:], linestyle="--", **kwargs)
