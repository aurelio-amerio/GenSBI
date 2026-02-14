# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
#
# --------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by Aurelio Amerio on 01-2026.
# Description: Ported implementation to use JAX instead of PyTorch.
# Fixed normalization bug where references were not scaled.
# Fixed histogram range bug.
# Reverted to single-pass implementation (no bootstrap).
# --------------------------------------------------------------------------

"""
Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

The TARP diagnostic is a global diagnostic which can be used to check a
trained posterior against a set of true values of theta.
"""

from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

from scipy.stats import kstest, norm, beta
import jax
from jax import numpy as jnp
from jax import Array

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from gensbi.diagnostics.metrics import l1, l2


@dataclass
class TARPResult:
    """
    Result of the TARP diagnostic.

    Parameters
    ----------
    ecp : Array
        Expected coverage probability. shape: (num_bootstrap, num_bins + 1) or (num_bins + 1,)
    alpha : Array
        Credibility values. shape: (num_bins + 1,)
    ecp_mean : Array
        Mean ECP. shape: (num_bins + 1,)
    ecp_lower : Optional[Array]
        Lower bound of ECP (e.g. 2.5% quantile). shape: (num_bins + 1,)
    ecp_upper : Optional[Array]
        Upper bound of ECP (e.g. 97.5% quantile). shape: (num_bins + 1,)
    """

    ecp: Array
    alpha: Array
    ecp_mean: Array
    ecp_lower: Optional[Array] = None
    ecp_upper: Optional[Array] = None

    @property
    def z_alpha(self) -> Array:
        """Z-scores corresponding to alpha (nominal coverage)."""
        return self._to_z(self.alpha)

    @property
    def z_mean(self) -> Array:
        """Z-scores corresponding to ecp_mean (empirical coverage)."""
        return self._to_z(self.ecp_mean)

    @property
    def z_lower(self) -> Optional[Array]:
        """Z-scores of the lower bound."""
        if self.ecp_lower is None:
            return None
        return self._to_z(self.ecp_lower)

    @property
    def z_upper(self) -> Optional[Array]:
        """Z-scores of the upper bound."""
        if self.ecp_upper is None:
            return None
        return self._to_z(self.ecp_upper)

    def _to_z(self, p: Array) -> Array:
        """Convert probability/credibility p to z-score."""
        p = np.array(p)
        eps = 1e-6
        p_clipped = np.clip(p, eps, 1 - eps)
        # Using 0.5 + p/2 puts 0.95 -> 0.975 -> 1.96
        # Using 1 - (1-p)/2 leads to same.
        return norm.ppf(0.5 + p_clipped / 2)


def alpha_from_z(z):
    """Convert z-score to alpha (significance level)."""
    return 2 - norm.cdf(z) * 2


def jefferys_interval(k, n, z=1.0):
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
    lower = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)

    # Handle edges manually if needed, though beta.ppf handles handles valid inputs
    # If k=0, lower=0. If k=n, upper=1.
    lower = np.where(k > 0, lower, 0.0)
    upper = np.where(k < n, upper, 1.0)

    return lower, upper


def run_tarp(
    thetas: Array,
    posterior_samples: Array,
    seed: int = 1,
    references: Optional[Array] = None,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = True,
    bootstrap: bool = False,
    num_bootstrap: int = 100,
) -> TARPResult:
    """
    Estimates coverage of samples given true values `thetas` with the TARP method.

    Reference
    ---------
    Lemos, Coogan et al. (2023). "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference". https://arxiv.org/abs/2302.03026

    Parameters
    ----------
    thetas : Array
        Ground-truth parameters for TARP, simulated from the prior. Shape: (num_tarp_samples, dim_theta).
    posterior_samples : Array
        Posterior samples. Shape: (num_posterior_samples, num_tarp_samples, dim_theta).
    seed : int, optional
        Random seed for sampling reference points. Default is 1.
    references : Array, optional
        Reference points for the coverage regions. If None, reference points are chosen uniformly from the parameter space.
    distance : Callable, optional
        Distance metric to use when computing the distance. Should accept two tensors and return distance values.
        Possible values: ``gensbi.diagnostics.metrics.l1`` or ``gensbi.diagnostics.metrics.l2``. ``l2`` is the default.
    num_bins : int, optional
        Number of bins to use for the credibility values. If None, then num_tarp_samples // 10 bins are used. Default is 30.
    z_score_theta : bool, optional
        Whether to normalize parameters before coverage test. Default is True.
    bootstrap : bool, optional
        Whether to use bootstrap to estimate uncertainties. Default is False.
    num_bootstrap : int, optional
        Number of bootstrap iterations to perform. Default is 100.

    Returns
    -------
    ecp : Array
        Expected coverage probability, see equation 4 of the paper.
    alpha : Array
        Credibility values, see equation 2 of the paper.
    """
    key = jax.random.PRNGKey(seed)

    # Ensure inputs are JAX arrays
    thetas = jnp.asarray(thetas)
    posterior_samples = jnp.asarray(posterior_samples)

    num_tarp_samples, dim_theta = thetas.shape

    num_posterior_samples = posterior_samples.shape[0]

    assert posterior_samples.shape == (
        num_posterior_samples,
        num_tarp_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for TARP: {posterior_samples.shape}, expected {(num_posterior_samples, num_tarp_samples, dim_theta)}"

    # Sample reference points uniformly if not provided
    # If bootstrapping, we regenerate references per iteration (defer to _run_tarp_bootstrap)
    # If not bootstrapping, we generate them once here.
    if references is None and not bootstrap:
        references = get_tarp_references(key, thetas)

    if references is not None:
        references = jnp.asarray(references)

    if not bootstrap:
        # If references was None, it's now generated (because not bootstrap)
        # If it was passed, it's asarray'd.
        ecp, alpha = _run_tarp_single(
            key,
            posterior_samples,
            thetas,
            references,
            distance,
            num_bins,
            z_score_theta,
        )

        # Calculate Jeffrey's intervals (95% CI -> z=1.96)
        num_sims = thetas.shape[0]
        k_values = np.array(ecp) * num_sims

        # Use z=1.96 for 95% CI to match plot_tarp labels
        lower, upper = jefferys_interval(k_values, num_sims, z=1.96)

        return TARPResult(
            ecp=ecp,
            alpha=alpha,
            ecp_mean=ecp,
            ecp_lower=jnp.array(lower),
            ecp_upper=jnp.array(upper),
        )

    # Bootstrap implementation
    # references might be None here (if it was None originally)
    ecp, alpha = _run_tarp_bootstrap(
        key,
        posterior_samples,
        thetas,
        references,
        distance,
        num_bins,
        z_score_theta,
        num_bootstrap,
    )

    # Compute statistics
    ecp_mean = jnp.mean(ecp, axis=0)
    ecp_lower = jnp.percentile(ecp, 2.5, axis=0)
    ecp_upper = jnp.percentile(ecp, 97.5, axis=0)

    return TARPResult(
        ecp=ecp,
        alpha=alpha,
        ecp_mean=ecp_mean,
        ecp_lower=ecp_lower,
        ecp_upper=ecp_upper,
    )


def _run_tarp_single(
    rng_key: Array,
    posterior_samples: Array,
    thetas: Array,
    references: Optional[Array],
    distance: Callable,
    num_bins: int,
    z_score_theta: bool,
) -> Tuple[Array, Array]:
    """Runs a single iteration of TARP."""
    if references is None:
        references = get_tarp_references(rng_key, thetas)

    return _compute_tarp(
        posterior_samples, thetas, references, distance, num_bins, z_score_theta
    )


def _run_tarp_bootstrap(
    rng_key: Array,
    posterior_samples: Array,
    thetas: Array,
    references: Optional[Array],
    distance: Callable,
    num_bins: int,
    z_score_theta: bool,
    num_bootstrap: int,
) -> Tuple[Array, Array]:

    num_sims = thetas.shape[0]

    # Define the bootstrap iteration
    def bootstrap_iter(key, _):
        # Split key for index sampling and potential reference generation
        rng_idx, rng_ref = jax.random.split(key)

        # Sample indices with replacement
        idx = jax.random.randint(rng_idx, shape=(num_sims,), minval=0, maxval=num_sims)

        # Resample data
        # Fix: index posterior_samples along the simulation axis (axis 1)
        boot_samples = posterior_samples[:, idx, :]
        boot_thetas = thetas[idx]

        # If references were not provided (None), generate them for this bootstrap sample
        if references is None:
            curr_references = get_tarp_references(rng_ref, boot_thetas)
        else:
            curr_references = references

        ecp, alpha = _compute_tarp(
            boot_samples,
            boot_thetas,
            curr_references,
            distance,
            num_bins,
            z_score_theta,
        )
        return ecp, alpha

    # VMAP approach
    # We need keys for each bootstrap
    keys = jax.random.split(rng_key, num_bootstrap)

    # vmap over keys
    ecp_results, alpha_results = jax.vmap(bootstrap_iter)(keys, None)

    # alpha should be the same for all (it depends on num_bins), so we just take the first one
    return ecp_results, alpha_results[0]


@jax.jit(static_argnames=["distance", "num_bins", "z_score_theta"])
def _compute_tarp(
    posterior_samples: Array,
    thetas: Array,
    references: Array,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = False,
) -> Tuple[Array, Array]:
    """
    Estimates coverage of samples given true values `thetas` with the TARP method.
    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape

    # Ensure num_bins is an integer (it might be passed as None in signature but handle it)
    # logic moved to run_tarp or ensured before calling JIT

    if z_score_theta:
        # Normalize all data to [0, 1] range based on theta bounds
        lo = thetas.min(axis=0, keepdims=True)  # min over batch
        hi = thetas.max(axis=0, keepdims=True)  # max over batch

        # Add epsilon to avoid division by zero
        denom = hi - lo + 1e-10

        posterior_samples = (posterior_samples - lo) / denom
        thetas = (thetas - lo) / denom
        # CRITICAL FIX: Normalize references using the same bounds
        references = (references - lo) / denom

    # distances between references and samples
    # Shape: (num_posterior_samples, num_tarp_samples)
    sample_dists = distance(references, posterior_samples)

    # distances between references and true values
    # Shape: (num_tarp_samples,)
    theta_dists = distance(references, thetas)

    # compute coverage, f in algorithm 2
    # Compare each posterior sample distance to the true theta distance
    # Broadcasting: (num_posterior_samples, num_tarp_samples) < (num_tarp_samples,)
    # We use vmap or broadcasting.
    # sample_dists: (n_post, n_sim)
    # theta_dists: (n_sim,)
    # comparison: (n_post, n_sim)

    coverage_values = (
        jnp.sum(sample_dists < theta_dists, axis=0) / num_posterior_samples
    )

    # CRITICAL FIX: Explicit range=(0.0, 1.0) ensures alpha grid is consistent
    hist, alpha_grid = jnp.histogram(
        coverage_values, density=True, bins=num_bins, range=(0.0, 1.0)
    )

    # calculate empirical CDF via cumsum and normalize
    ecp = jnp.cumsum(hist, axis=0) / hist.sum()

    # add 0 to the beginning of the ecp curve to match the alpha grid
    ecp = jnp.concatenate([jnp.zeros((1,)), ecp])

    return ecp, alpha_grid


def get_tarp_references(key, thetas: Array) -> Array:
    """Returns reference points for the TARP diagnostic, sampled from a uniform."""

    # obtain min/max per dimension of theta
    lo = thetas.min(axis=0)  # min for each theta dimension
    hi = thetas.max(axis=0)  # max for each theta dimension

    samples = jax.random.uniform(key, thetas.shape, minval=lo, maxval=hi)

    # sample one reference point for each entry in theta
    return samples


def check_tarp(
    result: TARPResult,
) -> Tuple[float, float]:
    r"""
    Check the obtained TARP credibility levels and expected coverage probabilities.

    Returns
    -------
    atc : float
        Area to curve, the difference between the ecp and alpha curve for :math:`\alpha > 0.5`.
    ks_prob : float
        p-value for a two-sample Kolmogorov-Smirnov test between ecp and alpha.
    """

    # Extract info from result
    ecp = result.ecp_mean
    alpha = result.alpha

    # get the index of the middle of the alpha grid
    midindex = alpha.shape[0] // 2
    # area to curve: difference between ecp and alpha above 0.5.
    atc = (ecp[midindex:] - alpha[midindex:]).sum().item()

    # Kolmogorov-Smirnov test between ecp and alpha
    kstest_pvals: float = kstest(np.array(ecp), np.array(alpha))[1]  # type: ignore

    return atc, kstest_pvals


def plot_tarp(
    result: TARPResult,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    mode: str = "both",
) -> Tuple[Figure, Union[Axes, Array]]:
    """
    Plot the expected coverage probability (ECP).

    Parameters
    ----------
    result : TARPResult
        Results from run_tarp.
    title : str, optional
        Title of the plot.
    figsize : tuple, optional
        Figure size.
    mode : str, optional
        "credibility", "confidence", or "both". Default is "credibility".
        "credibility" plots ECP vs Alpha.
        "confidence" plots Z-score(ECP) vs Z-score(Alpha).
    """

    if mode not in ["credibility", "confidence", "both"]:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'credibility', 'confidence', or 'both'."
        )

    if figsize is None:
        if mode == "both":
            figsize = (10, 4)
        else:
            figsize = (5, 4)

    fig = plt.figure(figsize=figsize)

    if mode == "both":
        ax = fig.subplots(1, 2)
        _plot_tarp_credibility(result, ax[0], title)
        _plot_tarp_confidence(result, ax[1], title)
    elif mode == "confidence":
        ax = plt.gca()
        _plot_tarp_confidence(result, ax, title)
    else:
        ax = plt.gca()
        _plot_tarp_credibility(result, ax, title)

    plt.tight_layout()
    return fig, ax  # type: ignore


def _plot_tarp_credibility(result: TARPResult, ax: Axes, title: Optional[str] = None):
    """Internal function to plot credibility mode."""
    ecp_mean = np.array(result.ecp_mean)
    alpha = np.array(result.alpha)

    # Plot mean
    ax.plot(alpha, ecp_mean, color="#202A44", label="TARP")

    # Plot bands if available
    if result.ecp_lower is not None and result.ecp_upper is not None:
        ax.fill_between(
            alpha,
            np.array(result.ecp_lower),
            np.array(result.ecp_upper),
            color="#202A44",
            alpha=0.2,
            label="95% CI",
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="darkgreen", label="Ideal")

    # Styling consistent with marginal_coverage
    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "TARP Coverage (Credibility)")
    # ax.set_aspect("equal")

    # Add Conservative / Overconfident text
    # ticks = [0.0, 1.0]
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)

    # compute the rotation angle
    x_ = [0, 10]
    y_ = [0, 10]
    p1 = ax.transData.transform((x_[0], y_[0]))
    p2 = ax.transData.transform((x_[1], y_[1]))
    dy_screen = p2[1] - p1[1]
    dx_screen = p2[0] - p1[0]
    phi = np.degrees(np.arctan2(dy_screen, dx_screen))
    # Positions similar to marginal_coverage but scaled to [0,1]
    ax.text(
        0.5,
        0.5 + 0.1,
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
        0.5 + 0.1,
        0.5,
        "Overconfident",
        ha="center",
        va="center",
        rotation=phi,
        color="darkgreen",
        alpha=1,
        rotation_mode="anchor",
        fontsize=11,
    )


def _plot_tarp_confidence(result: TARPResult, ax: Axes, title: Optional[str] = None):
    """Internal function to plot confidence mode (Z-scores)."""

    z_nominal = result.z_alpha
    z_empirical = result.z_mean

    bg_z_lower = result.z_lower
    bg_z_upper = result.z_upper

    # Plot mean
    ax.plot(z_nominal, z_empirical, color="#202A44", label="TARP")

    if bg_z_lower is not None and bg_z_upper is not None:
        ax.fill_between(
            z_nominal,
            bg_z_lower,
            bg_z_upper,
            color="#202A44",
            alpha=0.2,
            label="95% CI",
        )

    zmax = 3.5
    ax.plot([0, zmax], [0, zmax], "--", color="darkgreen", label="Ideal")

    ax.set_xlim(0, zmax)
    ax.set_ylim(0, zmax)
    ax.set_xlabel(r"Nominal coverage ($z$)")
    ax.set_ylabel(r"Empirical coverage ($\hat{z}$)")
    ax.set_title(title or "TARP Coverage (Confidence)")
    # ax.set_aspect("equal")

    # compute the rotation angle
    x_ = [0, 10]
    y_ = [0, 10]
    p1 = ax.transData.transform((x_[0], y_[0]))
    p2 = ax.transData.transform((x_[1], y_[1]))
    dy_screen = p2[1] - p1[1]
    dx_screen = p2[0] - p1[0]
    phi = np.degrees(np.arctan2(dy_screen, dx_screen))

    # Overconfident/Conservative
    ax.text(
        zmax / 2,
        (zmax / 2) * 1.2,
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
        (zmax / 2) * 1.2,
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

    # Add red dotted lines for 1, 2, 3 sigma levels
    for sigma in [1, 2, 3]:
        # Target Z (Nominal) is just sigma
        target_z = sigma

        # Interpolate achieved Z at this nominal Z
        if target_z <= z_nominal.max() and target_z >= z_nominal.min():
            achieved_z = np.interp(target_z, z_nominal, z_empirical)

            # Vertical line (Nominal)
            ax.plot([target_z, target_z], [0, achieved_z], ":", color="r", alpha=1)

            # Horizontal line (Empirical)
            ax.plot([0, target_z], [achieved_z, achieved_z], ":", color="r", alpha=1)

            # Convert sigma to percentage for label
            target_alpha = 2 * norm.cdf(target_z) - 1

            # Label sigma on X axis
            # marginal_coverage uses: ax.text(t, 0.3, ("%.2f%%" % (c * 100)), rotation=-90)
            ax.text(
                target_z + 0.02,
                0.1,
                f"{target_alpha:.2%}",
                color="k",
                ha="left",
                va="bottom",
                rotation=-90,
                # fontsize=9,
            )

            # Convert achieved Z to percentage for label
            achieved_p = 2 * norm.cdf(achieved_z) - 1

            # Label Z value on Y axis (Empirical Z converted to %)
            # marginal_coverage: ax.text(0.1, l + 0.05, ("%.2f%%" % (c*100)))
            ax.text(
                0.1,
                achieved_z + 0.02,
                f"{achieved_p:.2%}",
                color="k",
                ha="left",
                va="bottom",
                # fontsize=9,
            )

    # ax.legend(loc="upper left")
