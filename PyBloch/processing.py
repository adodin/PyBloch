""" Useful Functions for Processing Bloch Sphere Data

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np
import scipy.stats


def select_trajectories(x, y, z, n_plot):
    """ Selects a valid set of trajectories. Randomly chosen if n_plot is int.

    :param x ,y, z: (n_traj, n_time) array of Cartesian Bloch spehere coordinates
    :param n_plot: int or [int] how many (randomly chosen) or which trajectories to plot
    :return: x_plot, y_plot, z_plot n_plot trajectories
    """
    # Grab and validate trajectory indexes
    n_traj, n_time = x.shape
    assert x.shape == y.shape == z.shape
    if type(n_plot) is int:
        assert n_plot <= n_traj
        n_plot = np.random.choice(n_traj, n_plot, replace=False)
    else:
        assert max(n_plot) < n_traj

    # Select and return trajectories
    x_plot = np.array([x[i] for i in n_plot])
    y_plot = np.array([y[i] for i in n_plot])
    z_plot = np.array([z[i] for i in n_plot])

    return x_plot, y_plot, z_plot


def calculate_axes(x, y, z):
    """ Computes the instantaneous axes of rotation (as euler vectors) neglecting dephasing and dissipation

    :param x, y, z: (n_traj, n_time) array of Cartesian Bloch coordinates to be plotted (see select_trajectories)
    :return: (3, n_traj, n_time) array of Euler vector rotation axes. NB can be unpacked into u,v,w coordinates
    """

    # Combine coordinate into one array and compute derivative
    r = np.array([x, y, z])
    dr = np.gradient(r, axis=2)

    # Compute Cross Product to get axis of rotation and normalize
    rot = np.cross(r, dr, axis=0)
    norm = np.sqrt(np.sum(rot * rot, axis=0))
    rot = rot/norm

    # Calculate angle of rotation to linear order (small angle approximation
    theta = np.sum(r * dr,  axis=0)/np.sum(r * r, axis=0)

    return rot * theta


def norm_correlate(u, v, w, t_0=0):
    assert u.shape == v.shape == w.shape
    n_traj, n_time = u.shape

    rad = np.sqrt(u**2 + v**2 + w**2)
    rad_0 = rad[:, t_0]
    mu_0 = np.mean(rad_0)
    corr = []

    for i in range(n_time-t_0):
        rad_t = rad[:, t_0+i]
        mu_t = np.mean(rad_t)
        sig_0t = np.mean(rad_0*rad_t)
        corr.append(sig_0t - mu_t*mu_0)

    return np.array(corr)


def anisotropy_correlate(u, v, w, t_0=0):
    assert u.shape == v.shape == w.shape
    n_traj, n_time = u.shape

    r_0 = np.array([u[:, t_0], v[:, t_0], w[:, t_0]])
    mu_0 = np.mean(r_0, axis=-1)
    corr = []

    for i in range(n_time-t_0):
        r_t = np.array([u[:, t_0+i], v[:, t_0+i], w[:, t_0+i]])
        mu_t = np.mean(r_t, axis=-1)
        mu_0t = np.sum(mu_0 * mu_t)
        sig_0t = np.mean(np.sum(r_0 * r_t, axis =0))
        corr.append(sig_0t - mu_0t)

    return np.array(corr)


def gaussian_kde(x, y, z, h=None):
    """

    :param x, y, z: (n_sample) length set of cartesian coordinate samples
    :param h: KDE bandwidth. Use Scott's rule of thumb if None (default: None)
    :return: a callable density estimate. p_est([x_aarr,y_arr,z_arr]) returns values at sets of points
         see scipy.stats.gaussian_kde for info
    """
    return scipy.stats.gaussian_kde(np.array([x, y, z]), **kde_kwargs)


def evaluate_kde(x, y, z, p_est):
    """ Evaluates p_est on a grid defined by the arrays, x, y, z

    :param x, y, z: 1D Arrays or Mesh Grids defining grid points
    :param p_est:
    :return: distribution evaluated on grid if x,y,z are Mesh grids. Otherwise return X_grid, y_grid,z_grid , dist
    """
    assert x.shape == y.shape == z.shape
    # Checks if points are an array or a meshgrid
    is_array = len(x.shape) == 1

    # If points are arrays, convert to grids
    if is_array:
        x, y, z = np.meshgrid(x, y, z)

    # Flatten Mesh Grids into point list, evaluate and reshape into grids
    pts = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    dist = p_est(pts)
    dist = np.reshape(dist, x.shape)

    if is_array:
        return x, y, z, dist
    else:
        return dist
