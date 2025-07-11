import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import pandas as pd

sns.set_style("darkgrid")


def plot_trajectories(traj):
    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(traj[0,:,0], traj[0,:,1], color="red", s=1, alpha=1)
    ax.plot(traj[:,:,0], traj[:,:,1], color="white", lw=0.5, alpha=0.7)
    ax.scatter(traj[-1,:,0], traj[-1,:,1], color="blue", s=2, alpha=1, zorder=2)
    ax.set_aspect('equal', adjustable='box')
    # set black background
    ax.set_facecolor('#A6AEBF')
    plt.grid(False)
    return fig, ax

#plot marginals using seaborn's PairGrid

base_color = "#CD5656"  # Base color for the hexbin and kdeplot
hist_color = "#202A44"  # Color for the histograms 

rgb_base = np.array(mcolors.to_rgb(base_color))

colors = [(rgb_base[0], rgb_base[1], rgb_base[2], 0), # At data value 0, color is rgb_base with alpha 0
          (rgb_base[0], rgb_base[1], rgb_base[2], 1)] # At data value 1, color is rgb_base with alpha 1

transparent_cmap= LinearSegmentedColormap.from_list("transparent_red", colors, N=256)

def _parse_range(range_arg, ndim):
    if range_arg is None:
        return [None] * ndim
    if isinstance(range_arg, tuple) and len(range_arg) == 2 and not isinstance(range_arg[0], tuple):
        return [range_arg] * ndim
    if isinstance(range_arg, (list, tuple)) and len(range_arg) == ndim:
        return list(range_arg)
    raise ValueError("range must be a tuple (min, max) or a sequence of such tuples, one per axis")


def _plot_marginals_2d(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}, range=None, **kwargs):
    data = np.array(data)
    ndim = data.shape[1]
    fontsize=12
    if labels is None:
        labels = [r"$\theta_{}$".format(i) for i in np.arange(1, ndim + 1)]
    dataframe = pd.DataFrame(data, columns=labels)

    axis_ranges = _parse_range(range, ndim)
    xlim, ylim = axis_ranges[0], axis_ranges[1]

    cmap = hexbin_kwargs.pop('cmap', transparent_cmap)
    color = hexbin_kwargs.pop('color', [0,0,0,0])
    gridsize = hexbin_kwargs.pop('gridsize', gridsize)

    # Set extent for hexbin
    extent = None
    if xlim is not None and ylim is not None:
        extent = xlim + ylim
    joint_kws = dict(cmap=cmap, color=color, gridsize=gridsize, **hexbin_kwargs)
    if extent is not None:
        joint_kws['extent'] = extent

    marginal_kws = dict(bins=gridsize, fill=True, color=hist_color, **histplot_kwargs)
    if xlim is not None:
        marginal_kws['binrange'] = xlim
    if ylim is not None:
        marginal_kws['binrange'] = ylim

    g = sns.jointplot(
        data=dataframe, x=labels[0], y=labels[1], kind="hex", height=6, gridsize=gridsize,
        marginal_kws=marginal_kws, joint_kws=joint_kws, **kwargs)

    if xlim is not None:
        g.ax_joint.set_xlim(xlim)
        g.ax_marg_x.set_xlim(xlim)
    if ylim is not None:
        g.ax_joint.set_ylim(ylim)
        g.ax_marg_y.set_xlim(ylim)

    # Set fontsize for axis labels
    g.ax_joint.set_xlabel(labels[0], fontsize=fontsize)
    g.ax_joint.set_ylabel(labels[1], fontsize=fontsize)

    if plot_levels:
        levels = np.sort(1-np.array([0.6827, 0.9545]))
        g.plot_joint(sns.kdeplot, color=hist_color, zorder=3, levels=levels, alpha=1, linewidths=1)
    return g



def _plot_marginals_nd(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}, range=None, **kwargs):
    data = np.array(data)
    ndim = data.shape[1]
    fontsize=12

    if labels is None:
        labels = [r"$\theta_{}$".format(i) for i in np.arange(1, ndim + 1)]
    axis_ranges = _parse_range(range, ndim)
    cmap = hexbin_kwargs.pop('cmap', transparent_cmap)
    color = hexbin_kwargs.pop('color', [0,0,0,0])
    bins = histplot_kwargs.pop('bins', gridsize)
    fill = histplot_kwargs.pop('fill', True)
    color_hist = histplot_kwargs.pop('color', hist_color)

    fig, axes = plt.subplots(ndim, ndim, figsize=(2.5*ndim, 2.5*ndim))
    # Hide upper triangle and set all axes off by default
    for i in np.arange(ndim):
        for j in np.arange(ndim):
            if i < j:
                axes[i, j].set_visible(False)
            else:
                axes[i, j].set_visible(True)
            # Hide x/y ticks and labels for non-border plots
            if i != ndim-1:
                axes[i, j].set_xticklabels([])
                axes[i, j].set_xlabel("")
            if j != 0 and j != i:
                axes[i, j].set_yticklabels([])
                axes[i, j].set_ylabel("")

    # Lower triangle: hexbin and kde
    for i in np.arange(1, ndim):
        for j in np.arange(i):
            ax = axes[i, j]
            x = data[:, j]
            y = data[:, i]
            extent = None
            if axis_ranges[j] is not None and axis_ranges[i] is not None:
                extent = axis_ranges[j] + axis_ranges[i]
            ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, extent=extent, color=color, **hexbin_kwargs)
            if axis_ranges[j] is not None:
                ax.set_xlim(axis_ranges[j])
            if axis_ranges[i] is not None:
                ax.set_ylim(axis_ranges[i])
            if plot_levels:
                levels = np.sort(1-np.array([0.6827, 0.9545]))
                sns.kdeplot(x=x, y=y, levels=levels, color=hist_color, zorder=3, alpha=1, linewidths=1, ax=ax)
            # Only set axis labels for border plots
            if i == ndim-1:
                ax.set_xlabel(labels[j], fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=fontsize)

    # Diagonal: histograms
    for i in np.arange(ndim):
        ax = axes[i, i]
        x = data[:, i]
        binrange = axis_ranges[i] if axis_ranges[i] is not None else None
        sns.histplot(x, bins=bins, color=color_hist, fill=fill, binrange=binrange, ax=ax, stat='density', **histplot_kwargs)
        if axis_ranges[i] is not None:
            ax.set_xlim(axis_ranges[i])
        ax.autoscale(enable=True, axis='y', tight=False)
        # Only set y label for the top-left diagonal plot (theta_1)
        if i == 0:
            ax.set_ylabel(labels[i], fontsize=fontsize)
        else:
            ax.set_ylabel("")
        # Only set x label for bottom-right diagonal plot
        if i == ndim-1:
            ax.set_xlabel(labels[i], fontsize=14)
        else:
            ax.set_xlabel("")

    plt.tight_layout()
    return fig, axes


def plot_marginals(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}, range=None, **kwargs):
    if data.shape[1] == 2:
        return _plot_marginals_2d(data, plot_levels=plot_levels, labels=labels, gridsize=gridsize, hexbin_kwargs=hexbin_kwargs, histplot_kwargs=histplot_kwargs, range=range, **kwargs)
    else:
        return _plot_marginals_nd(data, plot_levels=plot_levels, labels=labels, gridsize=gridsize, hexbin_kwargs=hexbin_kwargs, histplot_kwargs=histplot_kwargs, range=range, **kwargs)