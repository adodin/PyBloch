""" Bloch Sphere Static Plotting in MayaVI

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np
from mayavi import mlab
import PyBloch.formatting


def save_fig(fname, **kwargs):
    """ Saves figure if fname is provided

    :param fname: str or None Target file name to save to. Does not save if none
    :param kwargs: kwargs for mlab savefig function
    :return:
    """
    if fname is not None:
        print("Saving Figure: " + fname)
        mlab.savefig(fname, **kwargs)


def select_trajectories(x, y, z, n_plot):
    """ Selects a valid set of trajectories. Randomly chosen if n_plot is int.

    :param x ,y, z: (n_traj, n_time) array of Cartesian Bloch spehere coordinates
    :param n_plot: int or [int] how many (randomly chosen) or which trajectories to plot
    :return: x_plot, y_plot, z_plot n_plot trajectories
    """
    # Grab and validate trajectory indexes
    n_traj, n_time = x.shape
    assert x.shape == y.shape == z.shape
    assert n_plot is int or n_plot[0] is int
    if n_plot is int:
        assert n_plot <= n_traj
        n_plot = np.random.choice(n_traj, n_plot, replace=False)
    else:
        assert max(n_plot) < n_traj

    # Select and return trajectories
    x_plot = np.array([x[i] for i in n_plot])
    y_plot = np.array([y[i] for i in n_plot])
    z_plot = np.array([z[i] for i in n_plot])

    return x_plot, y_plot, z_plot


def bloch_plot(x, y, z, frame, plot_traj=True, fname=None, show_fig=True, view=[30, 60, 10., (0, 0, 0)],
               fig_kwargs={'bgcolor': (1, 1, 1), 'fgcolor':(0, 0, 0)},
               mesh_kwargs={'color': (0.5, 0.5, 0.5), 'opacity': 0.25, 'tube_radius': 0.01},
               scatter_kwargs={}, line_kwargs={}, save_kwargs={}):
    """ Function for Plotting a set of states or trajectories on the Bloch Sphere

    :param x, y, z: (n_traj, n_time) array of Cartesian Bloch coordinates to be plotted (see select_trajectories)
    :param frame: int Frame to Plot
    :param plot_traj: boolean show trajectory leading up to state as a line
    :param fname: str or None Target file name
    :param show_fig: boolean show figure or not
    :param fig_kwargs: kwargs for mlab figure function
    :param axis_kwargs: kwargs for mlab axis fucntion
    :param scatter_kwargs: kwargs for mlab points3d function
    :param line_kwargs: kwargs for mlab plot3d function
    :return: scatter points handle
    """
    # Grab Number of Trajectories and Frames
    assert x.shape == y.shape == z.shape
    n_traj, n_time = x.shape
    assert frame < n_time

    # Make MLab Figure
    fig = mlab.figure(**fig_kwargs)

    # Draw Sphere Mesh
    lines = PyBloch.formatting.make_sphere(1.)
    mlab.view(30, 60, 10., (0, 0, 0))
    for l in lines:
        xl, yl ,zl = l
        mlab.plot3d(xl, yl, zl, **mesh_kwargs)

    # Add Points
    pts = mlab.points3d(x[:, frame], y[:, frame], z[:, frame], z[:, frame], vmin=-1, vmax=1, **scatter_kwargs)

    # Add Trajectories if Plotting
    if plot_traj:
        trajs = []
        for xt, yt, zt in zip(x, y, z):
            trajs.append(mlab.plot3d(xt[:frame+1], yt[:frame+1], zt[:frame+1], zt[:frame+1], vmin=-1, vmax=1, **line_kwargs))
        # Position Camera and adds Orientation Axes
        mlab.orientation_axes()
        save_fig(fname, **save_kwargs)
        if show_fig:
            mlab.show()
        return pts, trajs
    else:
        # Position Camera and adds Orientation Axes
        mlab.orientation_axes()
        save_fig(fname, **save_kwargs)
        if show_fig:
            mlab.show()
        return pts

def update_bloch(x, y, z, frame, pts, trajs=None):
    # Update Scatter Points
    pts.mlab_source.set(x=x[:, frame], y=y[:, frame], z=z[:, frame], scalars=z[:,frame])

    # Update Trajectories if used
    if trajs is not None:
        for traj, xt, yt, zt in zip(trajs, x, y, z):
            traj.mlab_source.reset(x=xt[:frame+1], y=yt[:frame+1], z=zt[:frame+1], scalars=zt[:frame+1])
