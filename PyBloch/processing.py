""" Useful Functions for Processing Bloch Sphere Data

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np


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
