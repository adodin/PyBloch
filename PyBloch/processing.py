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
