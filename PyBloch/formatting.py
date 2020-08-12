""" Useful Formatting Standards for MatPlotLib Visualisations
Written by: Amro Dodin (Willard Group - MIT)
"""

import numpy as np

red = [0.988, 0.078, 0.145]
blue = [0.008, 0.475, 1.]
green = [0.439, 0.678, 0.278]


def format_3d_axes(ax, x_label='x', y_label='y', z_label='z', x_lim=(-1., 1.), y_lim=(-1., 1.), z_lim=(-1., 1.),
                   x_ticks=3, y_ticks=3, z_ticks=3, pad=30, labelpad=90, font_kwargs={'name': 'serif', 'size': 40}):
    """ Single Line function for formatting 3D axes for Bloch sphere visualization

    :param ax: target axis handle
    :param x_label: Axis Labels
    :param y_label:
    :param z_label:
    :param x_lim: Axis Limits
    :param y_lim:
    :param z_lim:
    :param x_ticks: Axis Ticks
    :param y_ticks:
    :param z_ticks:
    :param pad: Axis units padding
    :param labelpad: Axis Label Padding
    :param font_kwargs: MatPlotLib font formatting dict
    :return: formatted axis handle (ax)
    """
    ax.set_xlim(x_lim)
    ax.tick_params(axis='x', pad=0.5*pad)
    ax.set_xticks(np.linspace(x_lim[0], x_lim[1], x_ticks))
    ax.set_xticklabels(np.linspace(x_lim[0], x_lim[1], x_ticks), **font_kwargs)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='y', pad=1.5*pad)
    ax.set_yticks(np.linspace(y_lim[0], y_lim[1], y_ticks))
    ax.set_yticklabels(np.linspace(y_lim[0], y_lim[1], y_ticks), **font_kwargs)
    ax.set_zlim(z_lim)
    ax.tick_params(axis='z', pad=pad)
    ax.set_zticks(np.linspace(z_lim[0], z_lim[1], z_ticks))
    ax.set_zticklabels(np.linspace(z_lim[0], z_lim[1], z_ticks), **font_kwargs)
    ax.set_xlabel(x_label, labelpad=labelpad, **font_kwargs)
    ax.set_ylabel(y_label, labelpad=labelpad, **font_kwargs)
    ax.set_zlabel(z_label, labelpad=labelpad, **font_kwargs)
    return ax


def make_sphere(rr, center=(0, 0, 0), n_mesh=25):
    """ Generates a spherical mesh of longitudinal lines

    :param rr: Sphere radius
    :param center: center position (e.g. tuple or array)
    :param n_mesh: number of lines in mesh
    :return: list of mesh lines
    """
    theta = np.linspace(0, np.pi, n_mesh)
    phi = np.linspace(-np.pi, np.pi, n_mesh)

    theta_grid, phi_grid = np.meshgrid(phi, theta)

    x_grid = rr*np.sin(theta_grid)*np.cos(phi_grid) + center[0]
    y_grid = rr*np.sin(theta_grid)*np.sin(phi_grid) + center[1]
    z = rr*np.cos(theta_grid) + center[2]

    # Creating the plot
    lines = []
    for i, j, k in zip(x_grid, y_grid, z):
        lines.append((i, j, k))
    for i, j, k in zip(x_grid.T, y_grid.T, z.T):
        lines.append((i, j, k))
    return lines[0:-1]
