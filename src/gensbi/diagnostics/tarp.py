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

from scipy.stats import kstest, norm
import jax
from jax import numpy as jnp
from jax import Array

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gensbi.diagnostics.metrics import l1, l2
from gensbi.diagnostics.utils import alpha_from_z, jefferys_interval, probit


@dataclass
class TARPResult:
    """
    Result of the TARP diagnostic.

    Stores the Expected Coverage Probability (ECP) curve and its uncertainty
    bounds. Provides z-score properties for the confidence-level view.

    Parameters
    ----------
    ecp : Array
        Expected coverage probability. shape: (num_bootstrap, num_bins + 1) or (num_bins + 1,)
    alpha : Array
        Credibility levels (histogram bin edges). shape: (num_bins + 1,)
    ecp_mean : Array
        Mean ECP across bootstrap iterations (or identical to ecp if no bootstrap).
    ecp_lower : Optional[Array]
        Lower bound of ECP (2.5th percentile or Jeffrey's lower). shape: (num_bins + 1,)
    ecp_upper : Optional[Array]
        Upper bound of ECP (97.5th percentile or Jeffrey's upper). shape: (num_bins + 1,)
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
        """Convert coverage probability p to z-score via probit(0.5 + p/2).

        This maps p ∈ [0, 1] to z ∈ [0, ∞) such that p = 0.6827 → z = 1 (1σ),
        p = 0.9545 → z = 2 (2σ), etc. Values are clipped to [eps, 1-eps] to
        avoid infinities at the boundaries.
        """
        p = np.array(p)
        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        return probit(0.5 + p_clipped / 2)


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
    thetas = jnp.asarray(thetas)
    posterior_samples = jnp.asarray(posterior_samples)

    num_tarp_samples, dim_theta = thetas.shape

    num_posterior_samples = posterior_samples.shape[0]

    assert posterior_samples.shape == (
        num_posterior_samples,
        num_tarp_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for TARP: {posterior_samples.shape}, expected {(num_posterior_samples, num_tarp_samples, dim_theta)}"

    # Generate references once for non-bootstrap; bootstrap regenerates per iteration
    if references is None and not bootstrap:
        references = get_tarp_references(key, thetas)

    if references is not None:
        references = jnp.asarray(references)

    if not bootstrap:
        ecp, alpha = _run_tarp_single(
            key,
            posterior_samples,
            thetas,
            references,
            distance,
            num_bins,
            z_score_theta,
        )

        # Without bootstrap, approximate 95% CI using Jeffrey's interval
        # (Beta prior-based interval for binomial proportions). This is
        # cheaper than bootstrap and gives smooth, well-calibrated bands.
        num_sims = thetas.shape[0]
        k_values = np.array(ecp) * num_sims
        lower, upper = jefferys_interval(k_values, num_sims, z=1.96)

        return TARPResult(
            ecp=ecp,
            alpha=alpha,
            ecp_mean=ecp,
            ecp_lower=jnp.array(lower),
            ecp_upper=jnp.array(upper),
        )

    # With bootstrap, uncertainty is estimated from the percentiles
    # of the ECP distribution across bootstrap iterations.
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

    ecp_mean = jnp.mean(ecp, axis=0)
    ecp_lower = jnp.percentile(ecp, 2.5, axis=0)  # 95% CI
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
    """Bootstrap TARP: resample (theta, posterior) pairs with replacement.

    Each iteration draws a bootstrap sample, optionally generates new
    reference points, and runs _compute_tarp. Uses jax.lax.scan instead
    of vmap so only one iteration is materialized at a time, avoiding
    OOM when the dataset is large.
    """

    num_sims = thetas.shape[0]

    def bootstrap_step(carry, key):
        rng_idx, rng_ref = jax.random.split(key)
        idx = jax.random.randint(rng_idx, shape=(num_sims,), minval=0, maxval=num_sims)

        boot_samples = posterior_samples[:, idx, :]
        boot_thetas = thetas[idx]

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
        return carry, (ecp, alpha)

    keys = jax.random.split(rng_key, num_bootstrap)
    _, (ecp_results, alpha_results) = jax.lax.scan(bootstrap_step, None, keys)

    # alpha is identical across bootstrap iterations (depends only on num_bins)
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
    Core TARP computation (JIT-compiled).

    For each simulation i, computes f_i = fraction of posterior samples that
    are closer to the reference point than the true parameter theta_i
    (Algorithm 2, Eq. 4 in Lemos et al.). Under perfect calibration,
    f_i ~ Uniform(0, 1), so the empirical CDF of {f_i} should follow the
    diagonal.
    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape

    if z_score_theta:
        # Normalize all arrays to [0, 1] per dimension so that the distance
        # metric treats all dimensions equally. References must be normalized
        # with the same bounds as thetas.
        lo = thetas.min(axis=0, keepdims=True)
        hi = thetas.max(axis=0, keepdims=True)
        denom = hi - lo + 1e-10  # avoid division by zero for constant dimensions

        posterior_samples = (posterior_samples - lo) / denom
        thetas = (thetas - lo) / denom
        references = (references - lo) / denom

    # For each simulation i, compute f_i = |{s : d(r_i, s) < d(r_i, θ_i)}| / N_post.
    # f_i is the fraction of posterior samples closer to the reference than
    # the true parameter (Eq. 4 in Lemos et al.). Under perfect calibration,
    # f_i ~ Uniform(0, 1).
    sample_dists = distance(references, posterior_samples)  # (n_post, n_sim)
    theta_dists = distance(references, thetas)  # (n_sim,)
    coverage_values = (
        jnp.sum(sample_dists < theta_dists, axis=0) / num_posterior_samples
    )

    # Bin the coverage values and build the empirical CDF.
    # range=(0, 1) ensures a consistent alpha grid across runs.
    hist, alpha_grid = jnp.histogram(
        coverage_values, density=True, bins=num_bins, range=(0.0, 1.0)
    )

    # Cumulative sum gives ECP at each alpha bin edge.
    # Prepend 0 so ECP(0) = 0, matching the alpha_grid which includes 0.
    ecp = jnp.cumsum(hist, axis=0) / hist.sum()
    ecp = jnp.concatenate([jnp.zeros((1,)), ecp])

    return ecp, alpha_grid


def get_tarp_references(key, thetas: Array) -> Array:
    """Sample reference points uniformly from the bounding box of theta."""
    lo = thetas.min(axis=0)
    hi = thetas.max(axis=0)
    return jax.random.uniform(key, thetas.shape, minval=lo, maxval=hi)


def check_tarp(
    result: TARPResult,
) -> Tuple[float, float]:
    r"""
    Quantitative check of the TARP coverage curve.

    Returns
    -------
    atc : float
        Area To Curve for :math:`\alpha > 0.5`. Positive means conservative
        (ECP above diagonal), negative means overconfident.
    ks_prob : float
        Two-sample KS test p-value. Low values indicate the ECP differs
        significantly from the ideal diagonal.
    """

    ecp = result.ecp_mean
    alpha = result.alpha

    midindex = alpha.shape[0] // 2
    atc = (ecp[midindex:] - alpha[midindex:]).sum().item()

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
        "credibility" plots ECP vs alpha.
        "confidence" plots z(ECP) vs z(alpha).
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

    ax.plot(alpha, ecp_mean, color="#202A44", label="TARP")

    if result.ecp_lower is not None and result.ecp_upper is not None:
        ax.fill_between(
            alpha,
            np.array(result.ecp_lower),
            np.array(result.ecp_upper),
            color="#202A44",
            alpha=0.2,
            label="95% CI",
        )

    ax.plot([0, 1], [0, 1], "--", color="darkgreen", label="Ideal")

    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "TARP Coverage (Credibility)")

    # Compute rotation angle for diagonal labels
    p1 = ax.transData.transform((0, 0))
    p2 = ax.transData.transform((10, 10))
    phi = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
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

    # Compute rotation angle for diagonal labels
    p1 = ax.transData.transform((0, 0))
    p2 = ax.transData.transform((10, 10))
    phi = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

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

    for sigma in [1, 2, 3]:
        target_z = sigma
        if target_z <= z_nominal.max() and target_z >= z_nominal.min():
            achieved_z = np.interp(target_z, z_nominal, z_empirical)
            ax.plot([target_z, target_z], [0, achieved_z], ":", color="r", alpha=1)
            ax.plot([0, target_z], [achieved_z, achieved_z], ":", color="r", alpha=1)

            target_alpha = 2 * norm.cdf(target_z) - 1
            ax.text(
                target_z + 0.02,
                0.1,
                f"{target_alpha:.2%}",
                color="k",
                ha="left",
                va="bottom",
                rotation=-90,
            )

            achieved_p = 2 * norm.cdf(achieved_z) - 1
            ax.text(
                0.1,
                achieved_z + 0.02,
                f"{achieved_p:.2%}",
                color="k",
                ha="left",
                va="bottom",
            )
