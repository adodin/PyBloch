""" Animation Utilities with MayaVI

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np
from mayavi import mlab
import PyBloch.plotting as plt
import moviepy.editor as mpy


def bloch_animate(x, y, z, plot_traj=True, fname=None, duration=1, fps=15, view=[30, 60, 10, (0, 0, 0)],
               fig_kwargs={'bgcolor': (1, 1, 1), 'fgcolor':(0, 0, 0)},
               mesh_kwargs={'color': (0.5, 0.5, 0.5), 'opacity': 0.25, 'tube_radius': 0.01},
               scatter_kwargs={}, line_kwargs={}, save_kwargs={}):
    """ Function for Animating Trajectories on Bloch Sphere

    :param x, y, z: (n_traj, n_time) array of Cartesian Bloch coordinates to be plotted (see select_trajectories)
    :param frame: int Frame to Plot
    :param plot_traj: boolean show trajectory leading up to state as a line
    :param fname: str or None Target file name (extension determined from this file) NB requires codec in ffmpeg
    :param duration: float default 1 length of animation in seconds
    :param fps: float default 15 animation frame rate (frames per second)
    :param view: [phi_cam, theta_cam, r_cam, (x_focal, y_focal, z_focal)] giving camera position and focal point
    :param show_fig: boolean show figure or not
    :param fig_kwargs: kwargs for mlab figure function
    :param axis_kwargs: kwargs for mlab axis fucntion
    :param scatter_kwargs: kwargs for mlab points3d function
    :param line_kwargs: kwargs for mlab plot3d function
    :return: scatter points handle
    """
    mlab.options.offscreen = True
    # Determine Required FPS
    assert x.shape == y.shape == z.shape
    n_traj, n_time = x.shape
    pps = n_time/duration        # trajectory points per second
    if fps > pps:
        print("Warning: Desired Frame Rate > Simulation. Output Frame Rate set by Simulation")
        fps = pps

    # Determine if gif or video
    if fname[-3:]=='gif':
        is_gif = True
    else:
        is_gif = False

    # Initialize Plot and set Frame function
    if plot_traj:
        pts, trajs = plt.bloch_plot(x=x, y=y, z=z, frame=0, plot_traj=plot_traj, fname=None, show_fig=False, view=view,
                                    fig_kwargs=fig_kwargs, mesh_kwargs=mesh_kwargs, scatter_kwargs=scatter_kwargs,
                                    line_kwargs=line_kwargs, save_kwargs={})

        def make_frame(t):
            i = int(np.round(t * pps))  # Finds simulation frame consistent with float time
            plt.update_bloch(x, y, z, i, pts, trajs)
            return mlab.screenshot()
    else:
        pts = plt.bloch_plot(x=x, y=y, z=z, frame=0, plot_traj=plot_traj, fname=None,
                             show_fig=False, view=view,fig_kwargs=fig_kwargs, mesh_kwargs=mesh_kwargs,
                             scatter_kwargs=scatter_kwargs, line_kwargs=line_kwargs, save_kwargs={})

        def make_frame(t):
            i = int(np.round(t * pps))  # Finds simulation frame consistent with float time
            plt.update_bloch(x, y, z, i, pts)
            return mlab.screenshot()

    # Make Video Clip and Save to desired format
    animation = mpy.VideoClip(make_frame, duration=duration)
    if is_gif:
        animation.write_gif(fname, fps=fps, **save_kwargs)
    else:
        animation.write_videofile(fname, fps=fps, **save_kwargs)

