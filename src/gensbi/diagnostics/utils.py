"""Shared statistical utilities for diagnostic functions."""

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
    k = np.asarray(k)
    n = np.asarray(n)
    alpha = alpha_from_z(z=z)
    lower = scipy.stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = scipy.stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)

    lower = np.where(k > 0, lower, 0.0)
    upper = np.where(k < n, upper, 1.0)

    return lower, upper
