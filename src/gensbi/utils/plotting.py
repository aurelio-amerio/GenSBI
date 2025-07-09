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
    return fig, ax

#plot marginals using seaborn's PairGrid

base_color = "#CD5656"  # Base color for the hexbin and kdeplot
hist_color = "#202A44"  # Color for the histograms 

rgb_base = np.array(mcolors.to_rgb(base_color))

colors = [(rgb_base[0], rgb_base[1], rgb_base[2], 0), # At data value 0, color is rgb_base with alpha 0
          (rgb_base[0], rgb_base[1], rgb_base[2], 1)] # At data value 1, color is rgb_base with alpha 1

transparent_cmap= LinearSegmentedColormap.from_list("transparent_red", colors, N=256)

def _plot_marginals_2d(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}):
    if labels is None:
        labels = [r"$\theta_{}$".format(i) for i in range(1, data.shape[1] + 1)]
    dataframe = pd.DataFrame(data, columns=labels)

    cmap = hexbin_kwargs.pop('cmap', transparent_cmap)
    color = hexbin_kwargs.pop('color', [0,0,0,0])  # Default to transparent color
    gridsize = hexbin_kwargs.pop('gridsize', gridsize)


    g = sns.jointplot(data=dataframe, x=labels[0], y=labels[1], kind="hex", height=6, gridsize=gridsize, marginal_kws=dict(bins=gridsize, fill=True, color=hist_color),joint_kws=dict(cmap=cmap, color=color, gridsize=gridsize))

    if plot_levels:
        levels = np.sort(1-np.array([0.6827, 0.9545]))
        g.plot_joint(sns.kdeplot, color=hist_color, zorder=3, levels=levels, alpha=1, linewidths=1)
    return g

def _plot_marginals_nd(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}):
    if labels is None:
        labels = [r"$\theta_{}$".format(i) for i in range(1, data.shape[1] + 1)]
    dataframe = pd.DataFrame(data, columns=labels)

    g = sns.PairGrid(dataframe, corner=True)
    cmap = hexbin_kwargs.pop('cmap', transparent_cmap)
    color = hexbin_kwargs.pop('color', [0,0,0,0])  # Default to transparent color
    gridsize = hexbin_kwargs.pop('gridsize', gridsize)

    g.map_lower(plt.hexbin, gridsize=gridsize, cmap=cmap, color=color, **hexbin_kwargs)
    if plot_levels:
        levels = np.sort(1-np.array([0.6827, 0.9545]))
        g.map_lower(sns.kdeplot, levels=levels, color=hist_color, zorder=3, alpha=1, linewidths=1, **hexbin_kwargs)

    bins = histplot_kwargs.pop('bins', gridsize)
    fill = histplot_kwargs.pop('fill', True)
    color = histplot_kwargs.pop('color', hist_color)
    g.map_diag(sns.histplot, bins=bins, color=color, fill=fill, **histplot_kwargs)
    return g

def plot_marginals(data, plot_levels=True, labels=None, gridsize=15, hexbin_kwargs={}, histplot_kwargs={}):
    if data.shape[1] == 2:
        return _plot_marginals_2d(data, plot_levels=plot_levels, labels=labels, gridsize=gridsize, hexbin_kwargs=hexbin_kwargs, histplot_kwargs=histplot_kwargs)
    else:
        return _plot_marginals_nd(data, plot_levels=plot_levels, labels=labels, gridsize=gridsize, hexbin_kwargs=hexbin_kwargs, histplot_kwargs=histplot_kwargs)