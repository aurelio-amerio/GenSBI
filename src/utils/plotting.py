import numpy as np
import matplotlib.pyplot as plt


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

#make functions to plot a vector field