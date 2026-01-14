# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
#
# --------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by Aurelio Amerio on 01-2026.
# Description: Ported implementation to use JAX instead of PyTorch.
# --------------------------------------------------------------------------

"""
Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

The TARP diagnostic is a global diagnostic which can be used to check a
trained posterior against a set of true values of theta.
"""

from typing import Callable, Optional, Tuple

from scipy.stats import kstest
import jax
from jax import numpy as jnp
from jax import Array

import numpy as np

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, FigureBase
import matplotlib.pyplot as plt


from gensbi.diagnostics.metrics import l1, l2


def run_tarp(
    thetas: Array,
    posterior_samples: Array,
    seed: int = 1,
    references: Optional[Array] = None,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = True,
) -> Tuple[Array, Array]:
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

    Returns
    -------
    ecp : Array
        Expected coverage probability, see equation 4 of the paper.
    alpha : Array
        Credibility values, see equation 2 of the paper.
    """
    key = jax.random.PRNGKey(seed)

    num_tarp_samples, dim_theta = thetas.shape

    num_posterior_samples = posterior_samples.shape[0]

    assert posterior_samples.shape == (
        num_posterior_samples,
        num_tarp_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for TARP: {posterior_samples.shape}, expected {(num_posterior_samples, num_tarp_samples, dim_theta)}"

    # Sample reference points uniformly if not provided
    if references is None:
        references = get_tarp_references(key, thetas)

    return _run_tarp(
        posterior_samples, thetas, references, distance, num_bins, z_score_theta
    )


def _run_tarp(
    posterior_samples: Array,
    thetas: Array,
    references: Array,
    distance: Callable = l2,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = False,
) -> Tuple[Array, Array]:
    """
    Estimates coverage of samples given true values `thetas` with the TARP method.

    Reference
    ---------
    Lemos, Coogan et al. (2023). "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference". https://arxiv.org/abs/2302.03026

    Parameters
    ----------
    posterior_samples : Array
        Predicted parameter samples to compute the coverage of. Shape: (num_posterior_samples, num_tarp_samples, dim_theta).
    thetas : Array
        True parameter values. Shape: (num_tarp_samples, dim_theta).
    references : Array
        Reference points for the coverage regions. Shape: (num_tarp_samples, dim_theta).
    distance : Callable, optional
        Distance metric to use when computing the distance. Should accept two tensors and return distance values. 
        Possible values: ``gensbi.diagnostics.metrics.l1`` or ``gensbi.diagnostics.metrics.l2``. ``l2`` is the default.
    num_bins : int, optional
        Number of bins to use for the credibility values. If None, then num_tarp_samples // 10 bins are used. Default is 30.
    z_score_theta : bool, optional
        Whether to normalize parameters before coverage test. Default is False.

    Returns
    -------
    ecp : Array
        Expected coverage probability, see equation 4 of the paper.
    alpha : Array
        Grid of credibility values, see equation 2 of the paper.
    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape

    assert (
        references.shape == thetas.shape
    ), "references must have the same shape as thetas"

    if num_bins is None:
        num_bins = num_tarp_samples // 10

    if z_score_theta:
        lo = thetas.min(axis=0, keepdims=True)  # min over batch
        hi = thetas.max(axis=0, keepdims=True)  # max over batch
        posterior_samples = (posterior_samples - lo) / (hi - lo + 1e-10)
        thetas = (thetas - lo) / (hi - lo + 1e-10)

    # distances between references and samples
    sample_dists = distance(references, posterior_samples)

    # distances between references and true values
    theta_dists = distance(references, thetas)

    # compute coverage, f in algorithm 2
    coverage_values = (
        jnp.sum(sample_dists < theta_dists, axis=0) / num_posterior_samples
    )

    hist, alpha_grid = jnp.histogram(
        coverage_values, density=True, bins=num_bins
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
    ecp: Array,
    alpha: Array,
) -> Tuple[float, float]:
    r"""
    Check the obtained TARP credibility levels and expected coverage probabilities.

    This diagnostic helps to uncover underdispersed, well-covering, or overdispersed posteriors.

    Let :math:`\mathrm{ecp}` be the expected coverage probability computed with the TARP method, and :math:`\alpha` the credibility levels (second output of ``run_tarp``).

    The area to curve (ATC) is defined as:

    .. math::
        \mathrm{ATC} = \sum_{i: \alpha_i > 0.5} \left( \mathrm{ecp}_i - \alpha_i \right)

    where values close to zero indicate well-calibrated posteriors. Values larger than zero indicate overdispersed distributions (the estimated posterior is too wide), while values smaller than zero indicate underdispersed distributions (the estimated posterior is too narrow). This property can also indicate if the posterior is biased (see Figure 2 of the reference paper).

    A two-sample Kolmogorov-Smirnov test is performed between :math:`\mathrm{ecp}` and :math:`\alpha` to test the null hypothesis that both distributions are identical (produced by one common CDF). The p-value should be close to 1 for well-calibrated posteriors. Commonly, the null is rejected if p-value is below 0.05.

    Reference
    ---------
    Lemos, Coogan et al. (2023). "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference". https://arxiv.org/abs/2302.03026

    Parameters
    ----------
    ecp : array-like
        Expected coverage probabilities computed with the TARP method (first output of ``run_tarp``).
    alpha : array-like
        Credibility levels :math:`\alpha` (second output of ``run_tarp``).

    Returns
    -------
    atc : float
        Area to curve, the difference between the ecp and alpha curve for :math:`\alpha > 0.5`.
    ks_prob : float
        p-value for a two-sample Kolmogorov-Smirnov test between ecp and alpha.
    """

    # get the index of the middle of the alpha grid
    midindex = alpha.shape[0] // 2
    # area to curve: difference between ecp and alpha above 0.5.
    atc = (ecp[midindex:] - alpha[midindex:]).sum().item()

    # Kolmogorov-Smirnov test between ecp and alpha
    kstest_pvals: float = kstest(np.array(ecp), np.array(alpha))[1]  # type: ignore

    return atc, kstest_pvals


def plot_tarp(
    ecp: Array, alpha: Array, title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot the expected coverage probability (ECP) against the credibility level (alpha).

    Parameters
    ----------
    ecp : array-like
        Array of expected coverage probabilities.
    alpha : array-like
        Array of credibility levels.
    title : str, optional
        Title for the plot. Default is "".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """

    fig = plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()
    
    ecp = np.array(ecp)
    alpha = np.array(alpha)

    ax.plot(alpha, ecp, color="blue", label="TARP")
    ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "")
    ax.legend()
    return fig, ax  # type: ignore