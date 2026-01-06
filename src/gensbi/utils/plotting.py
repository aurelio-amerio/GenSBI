"""
Plotting utilities for GenSBI.

This module provides visualization functions for generative models, including
trajectory plots, marginal distributions, and 2D contour plots. Supports both
seaborn and corner-based plotting styles.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import pandas as pd

from corner import corner

sns.set_style("darkgrid")


def plot_trajectories(traj):
    """
    Plot trajectories showing the flow from source to target distribution.
    
    Parameters
    ----------
        traj: Trajectory data of shape (time_steps, n_samples, n_dims).
        
    Returns
    -------
        Tuple of (figure, axes) objects.
    """
    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(traj[0, :, 0], traj[0, :, 1], color="red", s=1, alpha=1)
    ax.plot(traj[:, :, 0], traj[:, :, 1], color="white", lw=0.5, alpha=0.7)
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color="blue", s=2, alpha=1, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    # set black background
    ax.set_facecolor("#A6AEBF")
    plt.grid(False)
    return fig, ax


# plot marginals using seaborn's PairGrid

base_color = "#CD5656"  # Base color for the hexbin and kdeplot
hist_color = "#202A44"  # Color for the histograms
true_val_color = "#687FE5"

rgb_base = np.array(mcolors.to_rgb(base_color))

colors = [
    (
        rgb_base[0],
        rgb_base[1],
        rgb_base[2],
        0,
    ),  # At data value 0, color is rgb_base with alpha 0
    (rgb_base[0], rgb_base[1], rgb_base[2], 1),
]  # At data value 1, color is rgb_base with alpha 1

transparent_cmap = LinearSegmentedColormap.from_list("transparent_red", colors, N=256)


def _parse_range(range_arg, ndim):
    if range_arg is None:
        res = [None] * ndim
    elif (
        isinstance(range_arg, tuple)
        and len(range_arg) == 2
        and all(isinstance(x, (int, float)) for x in range_arg)
    ):
        res = [range_arg] * ndim
    elif (
        isinstance(range_arg, (list, tuple))
        and len(range_arg) == ndim
        and all(
            isinstance(r, tuple)
            and len(r) == 2
            and all(isinstance(x, (int, float)) for x in r)
            for r in range_arg
        )
    ):
        res = list(range_arg)
    else:
        raise ValueError(
            "Range must be None, a tuple (min, max), or a sequence of such tuples, one per axis"
        )
    return res


# def _plot_marginals_2d(
#     data,
#     plot_levels=True,
#     labels=None,
#     gridsize=15,
#     hexbin_kwargs={},
#     histplot_kwargs={},
#     range=None,
#     true_param=None,
#     **kwargs,
# ):
#     data = np.array(data)
#     if true_param is not None:
#         true_param = np.array(true_param)
#     ndim = data.shape[1]
#     fontsize = 12
#     if labels is None:
#         labels = ["$\\theta_{{{}}}$".format(i) for i in np.arange(1, data.shape[1] + 1)]
#     dataframe = pd.DataFrame(data, columns=labels)

#     axis_ranges = _parse_range(range, ndim)
#     xlim, ylim = axis_ranges[0], axis_ranges[1]

#     cmap = hexbin_kwargs.pop("cmap", transparent_cmap)
#     color = hexbin_kwargs.pop("color", [0, 0, 0, 0])
#     gridsize = hexbin_kwargs.pop("gridsize", gridsize)

#     # Set extent for hexbin
#     extent = None
#     if xlim is not None and ylim is not None:
#         extent = xlim + ylim
#     joint_kws = dict(cmap=cmap, color=color, gridsize=gridsize, **hexbin_kwargs)
#     if extent is not None:
#         joint_kws["extent"] = extent

#     marginal_kws = dict(bins=gridsize, fill=True, color=hist_color, **histplot_kwargs)

#     g = sns.jointplot(
#         data=dataframe,
#         x=labels[0],
#         y=labels[1],
#         xlim=xlim,
#         ylim=ylim,
#         kind="hex",
#         height=6,
#         gridsize=gridsize,
#         marginal_kws=marginal_kws,
#         joint_kws=joint_kws,
#         **kwargs,
#     )

#     if xlim is not None:
#         g.ax_joint.set_xlim(xlim)
#         g.ax_marg_x.set_xlim(xlim)
#     if ylim is not None:
#         g.ax_joint.set_ylim(ylim)
#         g.ax_marg_y.set_ylim(ylim)

#     # Set fontsize for axis labels
#     g.ax_joint.set_xlabel(labels[0], fontsize=fontsize)
#     g.ax_joint.set_ylabel(labels[1], fontsize=fontsize)

#     if plot_levels:
#         levels = np.sort(1 - np.array([0.6827, 0.9545]))
#         g.plot_joint(
#             sns.kdeplot,
#             color=hist_color,
#             zorder=3,
#             levels=levels,
#             alpha=1,
#             linewidths=1,
#         )

#     # Plot true_param if provided
#     if true_param is not None:
#         g.ax_joint.scatter(
#             true_param[0],
#             true_param[1],
#             color=true_val_color,
#             marker="s",
#             s=100,
#             zorder=10,
#         )
#         g.ax_joint.axvline(
#             true_param[0], color=true_val_color, linestyle="-", linewidth=1.5, zorder=5
#         )
#         g.ax_joint.axhline(
#             true_param[1], color=true_val_color, linestyle="-", linewidth=1.5, zorder=5
#         )
#     return g


# def _plot_marginals_nd(
#     data,
#     plot_levels=True,
#     labels=None,
#     gridsize=15,
#     range=None,
#     hexbin_kwargs={},
#     histplot_kwargs={},
#     true_param=None,
# ):
#     data = np.array(data)
#     if true_param is not None:
#         true_param = np.array(true_param)

#     ndim = data.shape[1]
#     fontsize = 12

#     if labels is None:
#         labels = ["$\\theta_{{{}}}$".format(i) for i in np.arange(1, data.shape[1] + 1)]
#     axis_ranges = _parse_range(range, ndim)
#     cmap = hexbin_kwargs.pop("cmap", transparent_cmap)
#     color = hexbin_kwargs.pop("color", [0, 0, 0, 0])
#     bins = histplot_kwargs.pop("bins", gridsize)
#     fill = histplot_kwargs.pop("fill", True)
#     color_hist = histplot_kwargs.pop("color", hist_color)

#     fig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim))
#     # Hide upper triangle and set all axes off by default
#     for i in np.arange(ndim):
#         for j in np.arange(ndim):
#             if i < j:
#                 axes[i, j].set_visible(False)
#             else:
#                 axes[i, j].set_visible(True)
#             # Hide x/y ticks and labels for non-border plots
#             if i != ndim - 1:
#                 axes[i, j].set_xticklabels([])
#                 axes[i, j].set_xlabel("")
#             if j != 0 and j != i:
#                 axes[i, j].set_yticklabels([])
#                 axes[i, j].set_ylabel("")

#     # Lower triangle: hexbin and kde
#     for i in np.arange(1, ndim):
#         for j in np.arange(i):
#             ax = axes[i, j]
#             x = data[:, j]
#             y = data[:, i]
#             extent = None
#             if axis_ranges[j] is not None and axis_ranges[i] is not None:
#                 extent = axis_ranges[j] + axis_ranges[i]
#             ax.hexbin(
#                 x,
#                 y,
#                 gridsize=gridsize,
#                 cmap=cmap,
#                 extent=extent,
#                 color=color,
#                 **hexbin_kwargs,
#             )
#             if axis_ranges[j] is not None:
#                 ax.set_xlim(axis_ranges[j])
#             if axis_ranges[i] is not None:
#                 ax.set_ylim(axis_ranges[i])
#             if plot_levels:
#                 levels = np.sort(1 - np.array([0.6827, 0.9545]))
#                 sns.kdeplot(
#                     x=x,
#                     y=y,
#                     levels=levels,
#                     color=hist_color,
#                     zorder=3,
#                     alpha=1,
#                     linewidths=1,
#                     ax=ax,
#                 )
#             # Plot true_param if provided
#             if true_param is not None:
#                 ax.scatter(
#                     true_param[j],
#                     true_param[i],
#                     color=true_val_color,
#                     marker="s",
#                     s=50,
#                     zorder=10,
#                     label="True",
#                 )
#                 ax.axvline(
#                     true_param[j],
#                     color=true_val_color,
#                     linestyle="-",
#                     linewidth=1.5,
#                     zorder=5,
#                 )
#                 ax.axhline(
#                     true_param[i],
#                     color=true_val_color,
#                     linestyle="-",
#                     linewidth=1.5,
#                     zorder=5,
#                 )
#             # Only set axis labels for border plots
#             if i == ndim - 1:
#                 ax.set_xlabel(labels[j], fontsize=fontsize)
#             if j == 0:
#                 ax.set_ylabel(labels[i], fontsize=fontsize)

#     # Diagonal: histograms
#     for i in np.arange(ndim):
#         ax = axes[i, i]
#         x = data[:, i]
#         binrange = axis_ranges[i] if axis_ranges[i] is not None else None
#         sns.histplot(
#             x,
#             bins=bins,
#             color=color_hist,
#             fill=fill,
#             binrange=binrange,
#             ax=ax,
#             stat="density",
#             **histplot_kwargs,
#         )
#         if true_param is not None:
#             ax.axvline(
#                 true_param[i],
#                 color=true_val_color,
#                 linestyle="-",
#                 linewidth=1.5,
#                 zorder=5,
#             )
#         if axis_ranges[i] is not None:
#             ax.set_xlim(axis_ranges[i])
#         ax.autoscale(enable=True, axis="y", tight=False)
#         # Only set y label for the top-left diagonal plot (theta_1)
#         if i == 0:
#             ax.set_ylabel(labels[i], fontsize=fontsize)
#         else:
#             ax.set_ylabel("")
#         # Only set x label for bottom-right diagonal plot
#         if i == ndim - 1:
#             ax.set_xlabel(labels[i], fontsize=14)
#         else:
#             ax.set_xlabel("")

#     plt.tight_layout()
#     return fig, axes


def _plot_marginals_seaborn(
    data,
    plot_levels=True,
    labels=None,
    gridsize=15,
    range=None,
    hexbin_kwargs={},
    histplot_kwargs={},
    true_param=None,
):
    data = np.array(data)
    if true_param is not None:
        true_param = np.array(true_param)

    ndim = data.shape[1]
    fontsize = 12

    if labels is None:
        labels = [f"$\\theta_{{{i}}}$" for i in np.arange(1, data.shape[1] + 1)]
    axis_ranges = _parse_range(range, ndim)
    cmap = hexbin_kwargs.pop("cmap", transparent_cmap)
    color = hexbin_kwargs.pop("color", [0, 0, 0, 0])
    bins = histplot_kwargs.pop("bins", gridsize)
    fill = histplot_kwargs.pop("fill", True)
    color_hist = histplot_kwargs.pop("color", hist_color)

    grid_kw = {}
    if ndim == 2:
        grid_kw = {'width_ratios': [6, 1], 'height_ratios': [1, 6]}

    fig, axes = plt.subplots(
        ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim), gridspec_kw=grid_kw
    )

    # Hide upper triangle and set axis properties
    for i in np.arange(ndim):
        for j in np.arange(ndim):
            if i < j:
                axes[i, j].set_visible(False)
            if i != ndim - 1:
                axes[i, j].set_xticklabels([])
                axes[i, j].set_xlabel("")
            if j != 0 and j != i:
                axes[i, j].set_yticklabels([])
                axes[i, j].set_ylabel("")

    # Lower triangle: hexbin and kde
    for i in np.arange(1, ndim):
        for j in np.arange(i):
            ax = axes[i, j]
            x_data, y_data = data[:, j], data[:, i]
            
            extent = axis_ranges[j] + axis_ranges[i] if axis_ranges[j] and axis_ranges[i] else None
            
            ax.hexbin(x_data, y_data, gridsize=gridsize, cmap=cmap, extent=extent, color=color, **hexbin_kwargs)

            if axis_ranges[j]: ax.set_xlim(axis_ranges[j])
            if axis_ranges[i]: ax.set_ylim(axis_ranges[i])

            if plot_levels:
                levels = np.sort(1 - np.array([0.6827, 0.9545]))
                sns.kdeplot(x=x_data, y=y_data, levels=levels, color=hist_color, zorder=3, alpha=1, linewidths=1, ax=ax)

            if true_param is not None:
                ax.scatter(true_param[j], true_param[i], color=true_val_color, marker="s", s=50, zorder=10)
                ax.axvline(true_param[j], color=true_val_color, ls="-", lw=1.5, zorder=5)
                ax.axhline(true_param[i], color=true_val_color, ls="-", lw=1.5, zorder=5)

            if i == ndim - 1: ax.set_xlabel(labels[j], fontsize=fontsize)
            if j == 0: ax.set_ylabel(labels[i], fontsize=fontsize)

    # diagonal: histograms
    for i in np.arange(ndim):
        ax = axes[i, i]
        x_data = data[:, i]
        binrange = axis_ranges[i] if axis_ranges[i] else None
        
        # 1. Determine orientation once
        is_rotated = (ndim == 2 and i == 1)
        
        # 2. Set plot parameters based on orientation
        hist_params = {
            'bins': bins, 'color': color_hist, 'fill': fill, 
            'binrange': binrange, 'stat': "density", **histplot_kwargs
        }
        if is_rotated:
            hist_params['y'] = x_data
        else:
            hist_params['x'] = x_data

        # 3. Make a single, clean plot call
        sns.histplot(ax=ax, **hist_params)


        # 4. Handle limits and true values based on orientation
        if is_rotated:
            if true_param is not None:
                ax.axhline(true_param[i], color=true_val_color, ls="-", lw=1.5, zorder=5)
            if axis_ranges[i]: ax.set_ylim(axis_ranges[i])
            ax.autoscale(enable=True, axis="x", tight=False)
        else:
            if true_param is not None:
                ax.axvline(true_param[i], color=true_val_color, ls="-", lw=1.5, zorder=5)
            if axis_ranges[i]: ax.set_xlim(axis_ranges[i])
            ax.autoscale(enable=True, axis="y", tight=False)

        # 5. Handle labels with simplified logic
        ax.set_xlabel(""); ax.set_ylabel("") # Default: no labels
        ax.set_yticklabels([])
        if ndim > 2:
            # if i == 0: ax.set_ylabel(labels[i], fontsize=fontsize)
            if i == ndim - 1: ax.set_xlabel(labels[i], fontsize=fontsize)
        if i != ndim - 1:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if ndim == 2:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_xticklabels([])
    
    if ndim == 2:

        y_ticks = axes[0,0].get_yticks()
        y_ticks = y_ticks[y_ticks > 0]
        axes[0,0].set_yticks(y_ticks)

        x_ticks = axes[1,1].get_xticks()
        x_ticks = x_ticks[x_ticks > 0]
        axes[1,1].set_xticks(x_ticks)

        fig.subplots_adjust(hspace=0.03, wspace=0.03, left=0.12, right=0.98, top=0.98, bottom=0.12)

    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.06, right=0.98, top=0.98, bottom=0.06)
    return fig, axes


def _plot_marginals_corner(
    data,
    labels=None,
    gridsize=25,
    range=None,
    true_param=None,
    **kwargs,
):
    data = np.array(data)
    ndim = data.shape[1]
    if range is not None:
        range = _parse_range(range, ndim)
    if true_param is not None:
        true_param = np.array(true_param)

    if labels is None:
        labels = ["$\\theta_{{{}}}$".format(i) for i in np.arange(1, data.shape[1] + 1)]
    plt.clf()
    corner(
        data,
        truths=true_param,
        bins=gridsize,
        labels=labels,
        color=base_color,  # points and 1D hist color
        hist_kwargs={
            "color": hist_color,
            "edgecolor": "white",
            "lw": 1,
            "histtype": "barstacked",
        },
        truth_color=true_val_color,
        contour_kwargs={"colors": hist_color, "linewidths": 1},
        range=range,
        **kwargs,
    )
    return plt.gcf(), plt.gca()


def plot_marginals(
    data,
    backend="corner",
    plot_levels=True,
    labels=None,
    gridsize=15,
    hexbin_kwargs={},
    histplot_kwargs={},
    range=None,
    true_param=None,
    **kwargs,
):
    """
    Plot marginal distributions of multidimensional data using either the 'corner' or 'seaborn' backend.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_dim)
        The data to plot. Each row is a sample, each column a parameter.
    backend : str, default="corner"
        Which plotting backend to use. Options:
        - 'corner': Use the corner.py package for a classic corner plot.
        - 'seaborn': Use seaborn's jointplot (2D) or custom grid (ND) for marginals.
        The seaborn backend is slower, but will produce smoother plots with KDE contours.
    plot_levels : bool, default=True
        If True and using seaborn, plot 1- and 2-sigma KDE contours on off-diagonal plots. When using 'corner', levels are automatically computed.
    labels : list of str or None, default=None
        Axis labels for each parameter. If None, uses LaTeX-style $\theta_i$.
    gridsize : int, default=15
        Number of bins for hexbin/histogram (seaborn) or for corner plot.
    hexbin_kwargs : dict, default={}
        Additional keyword arguments for hexbin plots (seaborn backend only).
    histplot_kwargs : dict, default={}
        Additional keyword arguments for histogram plots (seaborn backend only).
    range : tuple or list of tuples or None, default=None
        Axis limits for each parameter, e.g. [(xmin, xmax), (ymin, ymax), ...].
    true_param : array-like, shape (n_dim,), default=None
        Ground truth parameter values to mark on the plots.
    **kwargs :
        Additional keyword arguments passed to the underlying plotting functions.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes objects
        The figure and axes containing the plot.

    Raises
    ------
    ValueError
        If an unknown backend is specified.

    Notes
    -----
    - For 'corner', the function uses the corner.py package and supports labels, gridsize, range, and true_param.
    - For 'seaborn', 2D data uses jointplot, higher dimensions use a custom grid of hexbin and histogram plots.
    """
    if backend == "corner":
        return _plot_marginals_corner(
            data,
            labels=labels,
            gridsize=gridsize,
            range=range,
            true_param=true_param,
            **kwargs,
        )
    elif backend == "seaborn":


        return _plot_marginals_seaborn(
                data,
                plot_levels=plot_levels,
                labels=labels,
                gridsize=gridsize,
                hexbin_kwargs=hexbin_kwargs,
                histplot_kwargs=histplot_kwargs,
                range=range,
                true_param=true_param,
                **kwargs,
            )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'corner' or 'seaborn'.")


# code to plot a 2D likelihood

cmap_lcontour = sns.cubehelix_palette(
    start=0.5, rot=-0.5, light=1.0, dark=0.2, as_cmap=True
)


def plot_2d_levels(x, y, Z, ax, levels=[0.6827, 0.9545], display_labels=False):
    """
    Plot 2D levels on a given axis.

    Parameters
    ----------
    x : array-like
        X values.
    y : array-like
        Y values.
    Z : array-like
        Z values corresponding to (x, y).
    ax : matplotlib Axes
        The axes to plot on.
    levels : list of float
        The contour levels to plot.
    """

    # --- 1. Prepare the data ---

    x = np.asarray(x)  # make sure we have numpy arrays
    y = np.asarray(y)  # make sure we have numpy arrays
    Z = np.asarray(Z)  # make sure we have numpy arrays

    # --- 2. Define Desired Area Levels ---
    # These are the fractions of the total volume you want to enclose.
    # For a probability distribution, these are often confidence levels.
    area_levels = levels

    # --- 3. Calculate Contour Levels (Z-values) from Areas ---
    # To find the z-values that enclose a certain area, we follow these steps:
    # a. Flatten the 2D Z array into a 1D list of all values.
    # b. Sort these values in descending order (from highest to lowest).
    z_flat_sorted = np.sort(Z.ravel())[::-1]

    # c. Calculate the cumulative sum of the sorted values. Each element in
    #    this array represents the sum of all preceding (higher) values.
    z_cumsum = np.cumsum(z_flat_sorted)

    # d. Normalize the cumulative sum by the total sum of all Z values.
    #    This converts the cumulative sum into a fraction of the total volume,
    #    ranging from 0 to 1.
    z_cumsum_normalized = z_cumsum / z_cumsum[-1]

    # e. Find the z-values that correspond to our desired area fractions.
    #    We use np.searchsorted to find the index where the normalized
    #    cumulative sum first exceeds our target area level.
    indices = np.searchsorted(z_cumsum_normalized, area_levels)
    z_levels = z_flat_sorted[indices]

    # The levels must be sorted in ascending order for matplotlib's contour functions.
    z_levels = np.sort(z_levels)

    # --- 4. Plot the Results ---

    # To create filled contours, we need to define the boundaries of each color.
    # We start at 0, use our calculated z_levels, and end at the max value.
    # contour_fill_levels = np.concatenate(([Z.min()], z_levels, [Z.max()]))

    # a. Plot the filled contours (contourf).

    # b. Plot the contour lines (contour) for clarity.
    #    These lines will clearly mark the boundaries of the enclosed areas.
    cnt = ax.contour(x, y, Z, levels=z_levels, colors=hist_color, linewidths=1.5)

    if display_labels:
        labels = {z: f"{int(a*100)}%" for z, a in zip(z_levels, np.flip(area_levels))}
        ax.clabel(cnt, levels=z_levels, inline=True, fontsize=10, fmt=labels)

    return


def plot_2d_dist_contour(
    x,
    y,
    Z,
    true_param=None,
    levels=[0.6827, 0.9545],
    cmap=cmap_lcontour,
    display_labels=False
):
    """
    Plot a 2D contour plot of a distribution.

    Parameters
    ----------
    x : array-like
        X values.
    y : array-like
        Y values.
    Z : array-like
        Z values corresponding to (x, y).
    levels : list or None, optional
        Contour levels to plot. If None, contours will not be plotted.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        The figure and axes containing the plot.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.asarray(x)  # make sure we have numpy arrays
    y = np.asarray(y)  # make sure we have numpy arrays
    Z = np.asarray(Z)  # make sure we have numpy arrays

    ax.contourf(x, y, Z, levels=20, cmap=cmap, vmin=0)

    if levels is not None:
        plot_2d_levels(x, y, Z, ax, levels=levels, display_labels=display_labels)

    if true_param is not None:
        ax.scatter(
            true_param[0], true_param[1], color=base_color, s=50, marker="s", zorder=10
        )
        ax.axvline(
            true_param[0], color=base_color, linestyle="-", linewidth=1.5, zorder=9
        )
        ax.axhline(
            true_param[1], color=base_color, linestyle="-", linewidth=1.5, zorder=9
        )

    # Set aspect ratio to equal for better visualization
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    return fig, ax
