# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
#
# --------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by Aurelio Amerio on 01-2026.
# Description: Ported implementation to use JAX instead of PyTorch.
# --------------------------------------------------------------------------

import warnings
from typing import Callable, Dict, List, Tuple, Union

from jax import Array
import jax.numpy as jnp
import jax

import numpy as np

from scipy.stats import kstest, uniform
from tqdm import tqdm

from gensbi.diagnostics.metrics import c2st


from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
)


from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureBase
from scipy.stats import binom, gaussian_kde, iqr


def run_sbc(
    thetas: Array,
    xs: Array,
    posterior_samples: Array,
    reduce_fns: Union[
        str,
        Callable[[Array, Array], Array],
        List[Callable[[Array, Array], Array]],
    ] = "marginals",
    show_progress_bar: bool = True,
    **kwargs,
) -> Tuple[Array, Array]:
    """Run simulation-based calibration (SBC) or expected coverage.

    Note: This function implements two versions of coverage diagnostics:

    - Setting ``reduce_fns = "marginals"`` performs SBC as proposed in Talts et al.
      (see https://arxiv.org/abs/1804.06788).
    - Setting ``reduce_fns = posterior.log_prob`` performs sample-based expected
      coverage as proposed in Deistler et al.
      (see https://arxiv.org/abs/2210.04815).

    Parameters
    ----------
    thetas : Array
        Ground-truth parameters for SBC, simulated from the prior.
    xs : Array
        Observed data for SBC, simulated from thetas.
    posterior_samples : Array
        Samples from the posterior. Shape: (num_posterior_samples, num_sbc_samples, dim_theta).
    reduce_fns : str or Callable or List[Callable], optional
        Function used to reduce the parameter space into 1D.
        Simulation-based calibration can be recovered by setting this to the
        string `"marginals"`. Sample-based expected coverage can be recovered
        by setting it to `posterior.log_prob` (as a Callable). Default is "marginals".
    show_progress_bar : bool, optional
        Whether to display a progress bar over SBC runs. Default is True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    ranks : Array
        Ranks of the ground truth parameters under the inferred posterior.
    dap_samples : Array
        Samples from the data-averaged posterior.
    """
    # Remove NaNs and infinities from the input data.

    num_sbc_samples, dim_theta = thetas.shape

    num_posterior_samples = posterior_samples.shape[0]
    
    # _validate_sbc_inputs(thetas, xs, num_sbc_samples, num_posterior_samples)

    assert posterior_samples.shape == (
        num_posterior_samples,
        num_sbc_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for SBC: {posterior_samples.shape}, expected ({num_posterior_samples}, {num_sbc_samples}, {dim_theta})"

    # Take a random draw from each posterior to get data-averaged posterior samples.
    dap_samples = posterior_samples[0, :, :]
    assert dap_samples.shape == (num_sbc_samples, thetas.shape[1]), "Wrong DAP shape."

    # Calculate ranks
    ranks = _run_sbc(thetas, xs, posterior_samples, reduce_fns, show_progress_bar)

    return ranks, dap_samples


def _validate_sbc_inputs(
    thetas: Array, xs: Array, num_sbc_samples: int, num_posterior_samples: int
) -> None:
    """Validate inputs for the SBC procedure.

    Parameters
    ----------
    thetas : Array
        Ground-truth parameters.
    xs : Array
        Observed data.
    num_sbc_samples : int
        Number of SBC samples.
    num_posterior_samples : int
        Number of posterior samples.
    """
    if num_sbc_samples < 100:
        warnings.warn(
            "Number of SBC samples should be on the order of 100s to give reliable "
            "results.",
            stacklevel=2,
        )

    if num_posterior_samples < 100:
        warnings.warn(
            "Number of posterior samples for ranking should be on the order "
            "of 100s to give reliable SBC results.",
            stacklevel=2,
        )

    if thetas.shape[0] != xs.shape[0]:
        raise ValueError("Unequal number of parameters and observations.")


def _run_sbc(
    thetas: Array,
    xs: Array,
    posterior_samples: Array,
    reduce_fns: Union[
        str,
        Callable[[Array, Array], Array],
        List[Callable[[Array, Array], Array]],
    ] = "marginals",
    show_progress_bar: bool = True,
) -> Array:
    """Calculate ranks for SBC or expected coverage.

    Parameters
    ----------
    thetas : Array
        Ground-truth parameters.
    xs : Array
        Observed data corresponding to thetas.
    posterior_samples : Array
        Samples from posterior distribution.
    reduce_fns : str or Callable or List[Callable], optional
        Functions to reduce parameter space to 1D. Default is "marginals".
    show_progress_bar : bool, optional
        Whether to show progress bar. Default is True.

    Returns
    -------
    Array
        Array of ranks for each parameter and reduction function.
    """
    num_sbc_samples = thetas.shape[0]

    # Construct reduce functions for SBC or expected coverage.
    reduce_fns = _prepare_reduce_functions(reduce_fns, thetas.shape[1])

    # Initialize ranks tensor.
    ranks = np.zeros((num_sbc_samples, len(reduce_fns)))

    # Iterate over all SBC samples and calculate ranks.
    for sbc_idx, (true_theta, x_i) in tqdm(
        enumerate(zip(thetas, xs, strict=False)),
        total=num_sbc_samples,
        disable=not show_progress_bar,
        desc=f"Calculating ranks for {num_sbc_samples} SBC samples",
    ):
        # For each reduce_fn (e.g., per marginal for SBC)
        for dim_idx, reduce_fn in enumerate(reduce_fns):
            # Rank posterior samples against true parameter, reduced to 1D
            ranks[sbc_idx, dim_idx] = (
                (
                    reduce_fn(posterior_samples[:, sbc_idx, :], x_i)
                    < reduce_fn(true_theta[None, ...], x_i)
                )
                .sum()
                .item()
            )

    return ranks


def _prepare_reduce_functions(
    reduce_fns: Union[
        str,
        Callable[[Array, Array], Array],
        List[Callable[[Array, Array], Array]],
    ],
    param_dim: int,
) -> List[Callable[[Array, Array], Array]]:
    """Prepare reduction functions for SBC analysis.

    Parameters
    ----------
    reduce_fns : str or Callable or List[Callable]
        Function(s) to reduce parameters to 1D.
    param_dim : int
        Dimensionality of parameter space.

    Returns
    -------
    List[Callable]
        List of callable reduction functions.
    """
    # For SBC, we simply take the marginals for each parameter dimension.
    if isinstance(reduce_fns, str):
        if reduce_fns != "marginals":
            raise ValueError(
                "`reduce_fn` must either be the string `marginals` or a Callable or a "
                "List of Callables."
            )
        return [eval(f"lambda theta, x: theta[:, {i}]") for i in range(param_dim)]

    if isinstance(reduce_fns, Callable):
        return [reduce_fns]

    return reduce_fns


def check_sbc(
    ranks: Array,
    prior_samples: Array,
    dap_samples: Array,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> Dict[str, Array]:
    """Return uniformity checks and data-averaged posterior checks for SBC.

    Parameters
    ----------
    ranks : Array
        Ranks for each SBC run and for each model parameter,
        shape (N, dim_parameters).
    prior_samples : Array
        N samples from the prior.
    dap_samples : Array
        N samples from the data-averaged posterior.
    num_posterior_samples : int, optional
        Number of posterior samples used for SBC ranking. Default is 1000.
    num_c2st_repetitions : int, optional
        Number of times C2ST is repeated to estimate robustness. Default is 1.

    Returns
    -------
    Dict[str, Array]
        Dictionary containing:
        - ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
          one for each dim_parameters.
        - c2st_ranks: C2ST accuracy between ranks and uniform baseline,
          one for each dim_parameters.
        - c2st_dap: C2ST accuracy between prior and DAP samples, single value.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            "You are computing SBC checks with less than 100 samples. These checks "
            "should be based on a large number of test samples theta_o, x_o. We "
            "recommend using at least 100.",
            stacklevel=2,
        )

    # Run uniformity checks
    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )

    # Compare prior and data-averaged posterior
    c2st_scores_dap = check_prior_vs_dap(prior_samples, dap_samples)

    return {
        "ks_pvals": ks_pvals,
        "c2st_ranks": c2st_ranks,
        "c2st_dap": c2st_scores_dap,
    }


def check_prior_vs_dap(prior_samples: Array, dap_samples: Array) -> Array:
    """Returns the C2ST accuracy between prior and data-averaged posterior samples.

    C2ST is calculated for each dimension separately.

    According to simulation-based calibration, the inference method is well-calibrated
    if the data-averaged posterior samples follow the same distribution as the prior,
    i.e., if the C2ST score is close to 0.5. If it is not, then this suggests that the
    inference method is not well-calibrated (see Talts et al, "Simulation-based
    calibration" for details).

    Parameters
    ----------
    prior_samples : Array
        Samples from the prior distribution.
    dap_samples : Array
        Samples from the data-averaged posterior.

    Returns
    -------
    Array
        Array of C2ST scores for each parameter dimension.
    """
    if prior_samples.shape != dap_samples.shape:
        raise ValueError("Prior and DAP samples must have the same shape")

    return jnp.array([
        c2st(s1[:, None], s2[:, None])
        for s1, s2 in zip(prior_samples.T, dap_samples.T, strict=False)
    ])


def check_uniformity_frequentist(ranks: Array, num_posterior_samples: int) -> Array:
    """Return p-values for uniformity of the ranks using Kolmogorov-Smirnov test.

    Parameters
    ----------
    ranks : Array
        Ranks for each SBC run and for each model parameter,
        shape (N, dim_parameters).
    num_posterior_samples : int
        Number of posterior samples used for SBC ranking.

    Returns
    -------
    Array
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
        one for each dim_parameters.
    """
    kstest_pvals = jnp.array(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=jnp.float32,
    )

    return kstest_pvals


def check_uniformity_c2st(
    ranks: Array, num_posterior_samples: int, num_repetitions: int = 1, seed: int = 1
) -> Array:
    """Return C2ST scores for uniformity of the ranks.

    Run a C2ST between ranks and uniform samples.

    Parameters
    ----------
    ranks : Array
        Ranks for each SBC run and for each model parameter,
        shape (N, dim_parameters).
    num_posterior_samples : int
        Number of posterior samples used for SBC ranking.
    num_repetitions : int, optional
        Repetitions of C2ST tests to estimate classifier variance. Default is 1.
    seed : int, optional
        Random seed. Default is 1.

    Returns
    -------
    Array
        c2st_ranks: C2ST accuracy between ranks and uniform baseline,
        one for each dim_parameters.
    """
    
    key = jax.random.PRNGKey(seed)
    
    # Run C2ST multiple times to estimate stability
    c2st_scores = np.zeros((num_repetitions, ranks.shape[1]))
    
    for rep in range(num_repetitions):
        for dim_idx, rks in enumerate(ranks.T):
            key, subkey = jax.random.split(key)
            uniform_samples = jax.random.uniform(
                subkey,
                shape=(rks.shape[0],),
                minval=0,
                maxval=num_posterior_samples,
            )
            c2st_scores[rep, dim_idx] = c2st(
                rks[:, None],
                uniform_samples[:, None],
            ).item()
    

    # Use variance over repetitions to estimate robustness of C2ST
    c2st_std = c2st_scores.std(0, ddof=0 if num_repetitions == 1 else 1)
    if (c2st_std > 0.05).any():
        warnings.warn(
            f"C2ST score variability is larger than 0.05: std={c2st_std}, "
            "result may be unreliable. Consider increasing the number of samples.",
            stacklevel=2,
        )

    # Return the mean over repetitions as C2ST score estimate
    return c2st_scores.mean(0)



# plotting utilities
def sbc_rank_plot(
    ranks: Union[Array, np.ndarray, List[Array], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Additional options can be passed via the kwargs argument, see _sbc_rank_plot.

    Parameters
    ----------
    ranks : Array or List[Array]
        Array of ranks to be plotted shape (num_sbc_runs, num_parameters), or
        list of Arrays when comparing several sets of ranks, e.g., set of ranks
        obtained from different methods.
    num_posterior_samples : int
        Number of posterior samples used for ranking.
    num_bins : int, optional
        Number of bins used for binning the ranks. Default is num_sbc_runs / 20.
    plot_type : str, optional
        Type of SBC plot, histograms ("hist") or empirical cdfs ("cdf"). Default is "cdf".
    parameter_labels : List[str], optional
        List of labels for each parameter dimension.
    ranks_labels : List[str], optional
        List of labels for each set of ranks.
    colors : List[str], optional
        List of colors for each parameter dimension, or each set of ranks.
    fig : Figure, optional
        Figure object to plot in.
    ax : Axes, optional
        Axis object to plot in.
    figsize : tuple, optional
        Dimensions of figure object.
    **kwargs
        Additional keyword arguments passed to _sbc_rank_plot.

    Returns
    -------
    fig : Figure
        Figure object.
    ax : Axes
        Axis object.
    """

    return _sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins,
        plot_type,
        parameter_labels,
        ranks_labels,
        colors,
        fig=fig,
        ax=ax,
        figsize=figsize,
        **kwargs,
    )


def _sbc_rank_plot(
    ranks: Union[Array, np.ndarray, List[Array], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_repeats: int = 50,
    line_alpha: float = 0.8,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.3,
    xlim_offset_factor: float = 0.1,
    num_cols: int = 4,
    params_in_subplots: bool = False,
    show_ylabel: bool = False,
    sharey: bool = False,
    fig: Optional[FigureBase] = None,
    legend_kwargs: Optional[Dict] = None,
    ax=None,  # no type hint to avoid hassle with pyright. Should be `array(Axes).`
    figsize: Optional[tuple] = None,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Parameters
    ----------
    ranks : Array or List[Array]
        Array of ranks to be plotted shape (num_sbc_runs, num_parameters), or
        list of Arrays when comparing several sets of ranks, e.g., set of ranks
        obtained from different methods.
    num_posterior_samples : int
        Number of posterior samples used for ranking.
    num_bins : int, optional
        Number of bins used for binning the ranks. Default is num_sbc_runs / 20.
    plot_type : str, optional
        Type of SBC plot, histograms ("hist") or empirical cdfs ("cdf"). Default is "cdf".
    parameter_labels : List[str], optional
        List of labels for each parameter dimension.
    ranks_labels : List[str], optional
        List of labels for each set of ranks.
    colors : List[str], optional
        List of colors for each parameter dimension, or each set of ranks.
    num_repeats : int, optional
        Number of repeats for each empirical CDF step (resolution). Default is 50.
    line_alpha : float, optional
        Alpha for cdf lines or histograms. Default is 0.8.
    show_uniform_region : bool, optional
        Whether to plot the region showing the cdfs expected under uniformity. Default is True.
    uniform_region_alpha : float, optional
        Alpha for region showing the cdfs expected under uniformity. Default is 0.3.
    xlim_offset_factor : float, optional
        Factor for empty space left and right of the histogram. Default is 0.1.
    num_cols : int, optional
        Number of subplot columns, e.g., when plotting ranks of many parameters. Default is 4.
    params_in_subplots : bool, optional
        Whether to show each parameter in a separate subplot, or all in one. Default is False.
    show_ylabel : bool, optional
        Whether to show ylabels and ticks. Default is False.
    sharey : bool, optional
        Whether to share the y-labels, ticks, and limits across subplots. Default is False.
    fig : Figure, optional
        Figure object to plot in.
    legend_kwargs : Dict, optional
        Kwargs for the legend.
    ax : Axes, optional
        Axis object, must contain as many sublpots as parameters or len(ranks).
    figsize : tuple, optional
        Dimensions of figure object, default (8, 5) or (len(ranks) * 4, 5).

    Returns
    -------
    fig : Figure
        Figure object.
    ax : Axes
        Axis object.
    """

    if isinstance(ranks, (Array, np.ndarray)):
        ranks_list = [ranks]
    else:
        assert isinstance(ranks, List)
        ranks_list = ranks
    for idx, rank in enumerate(ranks_list):
        assert isinstance(rank, (Array, np.ndarray))
        if isinstance(rank, Array):
            ranks_list[idx]: np.ndarray = rank.numpy()  # type: ignore

    plot_types = ["hist", "cdf"]
    assert plot_type in plot_types, (
        f"plot type {plot_type} not implemented, use one in {plot_types}."
    )

    if legend_kwargs is None:
        legend_kwargs = dict(loc="best", handlelength=0.8)

    num_sbc_runs, num_parameters = ranks_list[0].shape
    num_ranks = len(ranks_list)

    # For multiple methods, and for the hist plots plot each param in a separate subplot
    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for ranki in ranks_list:
        assert ranki.shape == ranks_list[0].shape, (
            "all ranks in list must have the same shape."
        )

    num_rows = int(np.ceil(num_parameters / num_cols))
    if figsize is None:
        figsize = (num_parameters * 4, num_rows * 5) if params_in_subplots else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i + 1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i + 1}" for i in range(num_ranks)]
    if num_bins is None:
        # Recommendation from Talts et al.
        num_bins = num_sbc_runs // 20

    # Plot one row subplot for each parameter, different "methods" on top of each other.
    if params_in_subplots:
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                num_rows,
                min(num_parameters, num_cols),
                figsize=figsize,
                sharey=sharey,
            )
            ax = np.atleast_1d(ax)  # type: ignore
        else:
            assert ax.size >= num_parameters, (
                "There must be at least as many subplots as parameters."
            )
            num_rows = ax.shape[0] if ax.ndim > 1 else 1
        assert ax is not None

        col_idx, row_idx = 0, 0
        for ii, ranki in enumerate(ranks_list):
            for jj in range(num_parameters):
                col_idx = jj if num_rows == 1 else jj % num_cols
                row_idx = jj // num_cols
                plt.sca(ax[col_idx] if num_rows == 1 else ax[row_idx, col_idx])

                if plot_type == "cdf":
                    _plot_ranks_as_cdf(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_repeats,
                        ranks_label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior ranks {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=jj == 0,
                        alpha=line_alpha,
                    )
                    if ii == 0 and show_uniform_region:
                        _plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs,
                            num_bins,
                            num_repeats,
                            alpha=uniform_region_alpha,
                        )
                elif plot_type == "hist":
                    _plot_ranks_as_hist(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_posterior_samples,
                        ranks_label=ranks_labels[ii],
                        color="firebrick" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=show_ylabel,
                        alpha=line_alpha,
                        xlim_offset_factor=xlim_offset_factor,
                    )
                    # Plot expected uniform band.
                    _plot_hist_region_expected_under_uniformity(
                        num_sbc_runs,
                        num_bins,
                        num_posterior_samples,
                        alpha=uniform_region_alpha,
                    )
                    # show legend only in first subplot.
                    if jj == 0 and ranks_labels[ii] is not None:
                        plt.legend(**legend_kwargs)

                else:
                    raise ValueError(
                        f"plot_type {plot_type} not defined, use one in {plot_types}"
                    )
                # Remove empty subplots.
        col_idx += 1
        while num_rows > 1 and col_idx < num_cols:
            ax[row_idx, col_idx].axis("off")
            col_idx += 1

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks_list[0]
        for jj in range(num_parameters):
            _plot_ranks_as_cdf(
                ranki[:, jj],  # type: ignore
                num_bins,
                num_repeats,
                ranks_label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel="posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            _plot_cdf_region_expected_under_uniformity(
                num_sbc_runs,
                num_bins,
                num_repeats,
                alpha=uniform_region_alpha,
            )
        # show legend on the last subplot.
        plt.legend(**legend_kwargs)

    return fig, ax  # pyright: ignore[reportReturnType]


def _plot_ranks_as_hist(
    ranks: np.ndarray,
    num_bins: int,
    num_posterior_samples: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: str = "firebrick",
    alpha: float = 0.8,
    show_ylabel: bool = False,
    num_ticks: int = 3,
    xlim_offset_factor: float = 0.1,
) -> None:
    """Plot ranks as histograms on the current axis.

    Parameters
    ----------
    ranks : np.ndarray
        SBC ranks in shape (num_sbc_runs, ).
    num_bins : int
        Number of bins for the histogram, recommendation is num_sbc_runs / 20.
    num_posterior_samples : int
        Number of posterior samples used for ranking.
    ranks_label : str, optional
        Label for the ranks, e.g., when comparing ranks of different methods.
    xlabel : str, optional
        Label for the current parameter.
    color : str, optional
        Histogram color, default from Talts et al. Default is "firebrick".
    alpha : float, optional
        Histogram transparency. Default is 0.8.
    show_ylabel : bool, optional
        Whether to show y-label "counts". Default is False.
    num_ticks : int, optional
        Number of ticks on the x-axis. Default is 3.
    xlim_offset_factor : float, optional
        Factor for empty space left and right of the histogram. Default is 0.1.
    """
    xlim_offset = int(num_posterior_samples * xlim_offset_factor)
    plt.hist(
        ranks,
        bins=num_bins,
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.ylabel("counts")
    else:
        plt.yticks([])

    plt.xlim(-xlim_offset, num_posterior_samples + xlim_offset)
    plt.xticks(np.linspace(0, num_posterior_samples, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_ranks_as_cdf(
    ranks: np.ndarray,
    num_bins: int,
    num_repeats: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_ylabel: bool = True,
    num_ticks: int = 3,
) -> None:
    """Plot ranks as empirical CDFs on the current axis.

    Parameters
    ----------
    ranks : np.ndarray
        SBC ranks in shape (num_sbc_runs, ).
    num_bins : int
        Number of bins for the histogram, recommendation is num_sbc_runs / 20.
    num_repeats : int
        Number of repeats of each CDF step, i.e., resolution of the eCDF.
    ranks_label : str, optional
        Label for the ranks, e.g., when comparing ranks of different methods.
    xlabel : str, optional
        Label for the current parameter.
    color : str, optional
        Line color for the cdf.
    alpha : float, optional
        Line transparency. Default is 0.8.
    show_ylabel : bool, optional
        Whether to show y-label "counts". Default is True.
    num_ticks : int, optional
        Number of ticks on the x-axis. Default is 3.
    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_repeats: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity on the current axis.

    Parameters
    ----------
    num_sbc_runs : int
        Number of SBC runs.
    num_bins : int
        Number of bins.
    num_repeats : int
        Number of repeats.
    alpha : float, optional
        Alpha for the region. Default is 0.2.
    color : str, optional
        Color for the region. Default is "gray".
    """

    # Construct uniform histogram.
    uni_bins = binom(num_sbc_runs, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    # Decrease value one in last entry by epsilon to find valid
    # confidence intervals.
    uni_bins_cdf[-1] -= 1e-9

    lower = [binom(num_sbc_runs, p=p).ppf(0.005) for p in uni_bins_cdf]
    upper = [binom(num_sbc_runs, p=p).ppf(0.995) for p in uni_bins_cdf]

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_repeats * num_bins),
        y1=np.repeat(lower / np.max(lower), num_repeats),
        y2=np.repeat(upper / np.max(upper), num_repeats),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )


def _plot_hist_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_posterior_samples: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity.

    Parameters
    ----------
    num_sbc_runs : int
        Number of SBC runs.
    num_bins : int
        Number of bins.
    num_posterior_samples : int
        Number of posterior samples.
    alpha : float, optional
        Alpha for the region. Default is 0.2.
    color : str, optional
        Color for the region. Default is "gray".
    """

    lower = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.005)
    upper = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.995)

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_posterior_samples, num_bins),
        y1=np.repeat(lower, num_bins),
        y2=np.repeat(upper, num_bins),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )
