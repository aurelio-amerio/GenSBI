import warnings

import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from tqdm import tqdm

from gensbi.diagnostics.utils import (
    alpha_from_z,
    estimate_hat_z,
    plot_coverage_curve,
    resolution_limit_z,
)


def _prepare_inputs(theta, posterior_samples):
    """Convert the inputs to numpy and check that their shapes agree."""
    theta = np.asarray(theta)
    posterior_samples = np.asarray(posterior_samples)

    if posterior_samples.ndim != 3:
        raise ValueError(
            "posterior_samples must have shape (N_samples, N_batch, D_dim), "
            f"got {posterior_samples.shape}."
        )
    expected_shape = posterior_samples.shape[1:]
    if theta.shape != expected_shape:
        raise ValueError(
            f"Wrong theta shape for marginal coverage: {theta.shape}, expected "
            f"{expected_shape} to match posterior_samples of shape {posterior_samples.shape}."
        )
    return theta, posterior_samples


def _randomised_rank(sample_scores, truth_score, u):
    """
    Fraction of the pooled points that score higher than the truth, ties split at random.

    With ``u ~ Uniform(0, 1)`` this is exactly Uniform(0, 1) whenever the truth
    and the samples are exchangeable and the scores are a symmetric function
    of the pooled set.
    """
    higher = np.count_nonzero(sample_scores > truth_score)
    tied = np.count_nonzero(sample_scores == truth_score)
    return (higher + u * (tied + 1)) / (sample_scores.size + 1)


def _histogram_scores(pooled, bins):
    """Score every pooled point by the count of the histogram bin it falls in."""
    counts, edges = np.histogram(pooled, bins=bins)
    idx = np.clip(np.searchsorted(edges, pooled, side="right") - 1, 0, len(counts) - 1)
    # Only possible with user-supplied edges: a point outside them is in no bin.
    inside = (pooled >= edges[0]) & (pooled <= edges[-1])
    return np.where(inside, counts[idx], 0)


def _kde_scores(pooled, grid_points, bw):
    """Score every pooled point by the kernel density estimate of the pooled set."""
    estimator = FFTKDE(bw=bw)  # raises here on an invalid bandwidth argument
    try:
        # The selector's own floating-point warnings precede a failure we handle.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            x_grid, y_grid = estimator.fit(pooled).evaluate(grid_points)
    except Exception:
        # Data-driven selectors such as ISJ fail on degenerate samples or when the
        # truth is extremely far from them. Those are cases we must still rank,
        # so fall back to a rule of thumb that always returns a bandwidth.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_grid, y_grid = FFTKDE(bw="silverman").fit(pooled).evaluate(grid_points)
    return np.interp(pooled, x_grid, y_grid)


def _compute_marginal_coverage(theta, posterior_samples, score_fn, seed, show_progress_bar):
    """
    Shared loop of the marginal coverage estimators.

    ``score_fn`` maps the pooled points (the posterior samples followed by the
    true value) to one density score per point.
    """
    theta, posterior_samples = _prepare_inputs(theta, posterior_samples)
    _, N_batch, D_dim = posterior_samples.shape

    # One tie-breaking draw per (dimension, observation), fixed up front so the
    # result does not depend on the order of evaluation.
    u = np.random.default_rng(seed).uniform(size=(D_dim, N_batch))

    alpha_values = np.full((D_dim, N_batch), np.nan)

    pairs = [(d, b) for d in range(D_dim) for b in range(N_batch)]
    for d, b in tqdm(
        pairs,
        desc="Computing marginal coverage",
        disable=not show_progress_bar,
        leave=False,
    ):
        # Pool the truth with the samples. If the posterior is calibrated the
        # N + 1 points are exchangeable, which is what makes the rank exact.
        pooled = np.append(posterior_samples[:, b, d], theta[b, d]).astype(np.float64)

        if not np.all(np.isfinite(pooled)):
            continue  # left as NaN and reported below

        scores = score_fn(pooled)
        alpha_values[d, b] = _randomised_rank(scores[:-1], scores[-1], u[d, b])

    num_failed = int(np.isnan(alpha_values).sum())
    if num_failed > 0:
        warnings.warn(
            f"The credibility could not be computed for {num_failed} of "
            f"{alpha_values.size} (dimension, observation) pairs, because their "
            "posterior samples or true value are not finite. They are set to NaN.",
            UserWarning,
            stacklevel=4,
        )

    return alpha_values


def _compute_marginal_coverage_KDE(
    theta,
    posterior_samples,
    grid_points=2048,
    bw="ISJ",
    seed=0,
    show_progress_bar=True,
):
    r"""
    Computes the marginal coverage (credibility) for each observation and dimension
    using Kernel Density Estimation.

    For each observation b and dimension d this estimates the HPD (Highest
    Posterior Density) credibility of the true value,

    .. math::
        \alpha = \int_{p(x) > p(\theta^*)} p(x) dx ,

    the posterior mass in regions denser than the true value. If the posterior
    is well calibrated, alpha is Uniform(0, 1).

    The true value is pooled with the posterior samples, a 1D KDE is fitted to
    the pooled points, and alpha is the fraction of samples whose density is
    higher than that of the true value, with ties split at random. See
    :func:`compute_marginal_coverage` for why the true value is pooled.

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
        (Improved Sheather-Jones, a plug-in bandwidth selector). If the
        selector fails on the pooled points, Silverman's rule is used instead.
    seed : int or numpy.random.Generator, optional
        Seed of the random tie-breaking. Default is 0.
    show_progress_bar : bool, optional
        Whether to display a progress bar. Default is True.

    Returns
    -------
    alpha : np.ndarray of shape (D_dim, N_batch)
        The credibility values, strictly between 0 and 1. NaN where the
        samples or the true value are not finite.
    """
    return _compute_marginal_coverage(
        theta,
        posterior_samples,
        lambda pooled: _kde_scores(pooled, grid_points, bw),
        seed,
        show_progress_bar,
    )


def _compute_marginal_coverage_histogram(
    theta, posterior_samples, bins="stone", seed=0, show_progress_bar=True
):
    """
    Computes marginal coverage using empirical histograms.

    This is a discrete approximation of the HPD credibility. For each
    (observation, dimension) pair the true value is pooled with the posterior
    samples and the pooled points are binned. Every point is scored by the
    count of its own bin, and alpha is the fraction of samples that score
    higher than the true value, with ties split at random. See
    :func:`compute_marginal_coverage` for why the true value is pooled.

    Faster than KDE and works well for multimodal distributions.

    Parameters
    ----------
    theta : array_like of shape (N_batch, D_dim)
        The ground truth parameters.
    posterior_samples : array_like of shape (N_samples, N_batch, D_dim)
        Samples from the posterior.
    bins : int, str or sequence of scalars, optional
        Number of bins, binning strategy or bin edges. See
        ``numpy.histogram_bin_edges`` for valid string values
        ('stone', 'scott', 'fd', 'sturges', etc). Default is 'stone'.
    seed : int or numpy.random.Generator, optional
        Seed of the random tie-breaking. Default is 0.
    show_progress_bar : bool, optional
        Whether to display a progress bar. Default is True.

    Returns
    -------
    alpha_values : np.ndarray of shape (D_dim, N_batch)
        The credibility values, strictly between 0 and 1. NaN where the
        samples or the true value are not finite.
    """
    return _compute_marginal_coverage(
        theta,
        posterior_samples,
        lambda pooled: _histogram_scores(pooled, bins),
        seed,
        show_progress_bar,
    )


def compute_marginal_coverage(
    theta,
    posterior_samples,
    method="histogram",
    seed=0,
    show_progress_bar=True,
    **kwargs,
):
    r"""
    Compute marginal coverage for each observation and dimension.

    For every observation and parameter dimension this returns the HPD
    (Highest Posterior Density) credibility of the true value: the posterior
    mass :math:`\alpha` in regions denser than the true value. For a
    calibrated posterior :math:`\alpha` is Uniform(0, 1), so the fraction of
    observations with :math:`\alpha \leq c` should equal :math:`c`
    (see :func:`plot_marginal_coverage`).

    Notes
    -----
    The density is estimated from the true value *pooled with* the posterior
    samples, and :math:`\alpha` is the randomised rank of the true value's
    density among the samples' densities. If the posterior is calibrated, the
    true value and the samples are exchangeable draws from the same
    distribution, so this rank is exactly uniform for any posterior shape,
    any number of samples and any binning or bandwidth.

    Ranking the true value against a density built from the samples alone is
    biased towards overconfidence: every sample raises the density estimate at
    its own location and the true value does not. That bias only decays as
    :math:`1/\sqrt{N_\mathrm{samples}}`.

    Tied densities are split at random, which makes the result depend on
    ``seed``. A flat marginal has no density ordering at all, so its
    credibility is pure noise: HPD coverage cannot test such a marginal, use
    SBC ranks (:func:`gensbi.diagnostics.run_sbc`) instead.

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
    seed : int or numpy.random.Generator, optional
        Seed of the random tie-breaking. Default is 0.
    show_progress_bar : bool, optional
        Whether to display a progress bar. Default is True.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the chosen method
        (``bins`` for "histogram"; ``grid_points`` and ``bw`` for "KDE").

    Returns
    -------
    alpha_values : np.ndarray of shape (D_dim, N_batch)
        The credibility values, strictly between 0 and 1. NaN where the
        samples or the true value are not finite; a warning reports how many.
    """
    if method == "histogram":
        estimator = _compute_marginal_coverage_histogram
    elif method == "KDE":
        estimator = _compute_marginal_coverage_KDE
    else:
        raise ValueError(f"Unknown method: {method}")
    return estimator(
        theta,
        posterior_samples,
        seed=seed,
        show_progress_bar=show_progress_bar,
        **kwargs,
    )


def plot_marginal_coverage(
    alpha_values,
    zmax=3.5,
    n_cols=3,
    figsize=None,
    parameter_labels=None,
    num_posterior_samples=None,
):
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
    parameter_labels : list of str, optional
        Title of each panel, one per parameter dimension. If None, the panels
        are titled "Dim 1", "Dim 2", ...
    num_posterior_samples : int, optional
        Number of posterior samples the credibilities were computed from. If
        given, the curve is dashed above the confidence level that this many
        samples can resolve (a tail mass of ``1 / (num_posterior_samples + 1)``,
        z = 3.3 for 1000 samples). The dashed part is an extrapolation.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """

    n_cols = min(n_cols, alpha_values.shape[0])

    D_dim, N_batch = alpha_values.shape

    if parameter_labels is None:
        parameter_labels = [f"Dim {d + 1}" for d in range(D_dim)]
    elif len(parameter_labels) != D_dim:
        raise ValueError(
            f"parameter_labels has {len(parameter_labels)} entries, "
            f"but alpha_values has {D_dim} dimensions."
        )

    n_rows = (D_dim + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)  # handle 1D case

    axes_flat = axes.flatten()

    z_resolved = None
    if num_posterior_samples is not None:
        z_resolved = resolution_limit_z(num_posterior_samples)

    for k in range(D_dim, len(axes_flat)):
        axes_flat[k].axis("off")

    for d in range(D_dim):
        ax = axes_flat[d]

        hat_z = estimate_hat_z(alpha_values[d], zmax=zmax)
        z_grid = hat_z["z"]

        plot_coverage_curve(ax, z_grid, hat_z["mean"], z_resolved, color="k")

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
        ax.set_title(parameter_labels[d])

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
