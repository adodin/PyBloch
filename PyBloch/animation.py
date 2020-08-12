""" Animation Utilities with MayaVI

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np
from mayavi import mlab
import PyBloch.plotting as plt
import moviepy.editor as mpy


def bloch_animate(x, y, z, plot_traj=True, fname=None, duration=1,
               fig_kwargs={'bgcolor': (1, 1, 1), 'fgcolor':(0, 0, 0)},
               mesh_kwargs={'color': (0.5, 0.5, 0.5), 'opacity': 0.25, 'tube_radius': 0.01},
               scatter_kwargs={}, line_kwargs={}, save_kwargs={}):
    """ Function for Animating Trajectories on Bloch Sphere

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
    # Determine Required FPS
    assert x.shape == y.shape == z.shape
    n_traj, n_time = x.shape
    fps = n_time/duration

    # Initialize Plot and set Frame function
    if plot_traj:
        pts, trajs = plt.bloch_plot(x=x, y=y, z=z, frame=0, plot_traj=plot_traj, fname=None, show_fig=False,
                                    fig_kwargs=fig_kwargs, mesh_kwargs=mesh_kwargs, scatter_kwargs=scatter_kwargs,
                                    line_kwargs=line_kwargs, save_kwargs={})

        def make_frame(t):
            plt.update_bloch(x, y, z, int(np.round(t*fps)), pts, trajs)
            return mlab.screenshot()
    else:
        pts = plt.bloch_plot(x, y, z, 0, plot_traj, None, False,
                                    fig_kwargs, mesh_kwargs, scatter_kwargs, line_kwargs, save_kwargs)

        def make_frame(t):
            plt.update_bloch(x, y, z, int(np.round(t*fps)), pts)
            return mlab.screenshot()


    # Make Video Clip and Save as Gif
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif(fname, fps=fps)
