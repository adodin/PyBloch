""" Functions for Sampling Initial Conditions

Written By: Amro Dodin (Willard Group - MIYT)
"""

import numpy as np
import numpy.random as rnd
import numpy.linalg as la


def sample_magnitudes(num_samples, dim):
    """ Samples Normalized Magnitudes for Wave-Functions

    :param num_samples: Number of Wavefunctions to Sample
    :param dim: Hilbert Space Dimension
    :return: (num_samples, dim) array of wavefunction magnitudes
    """

    mags = rnd.uniform(0, 1, (num_samples, dim))
    mags = mags / la.norm(mags, axis=1, keepdims=True)
    return mags


def sample_phases(num_samples, dim):
    """ Samples Phase Factors for Wave-Functions Uniformly

    :param num_samples: Number of Wavefunctions to Sample
    :param dim: Hilbert Space Dimension
    :return: (num_samples, dim) array of wavefunction magnitudes
    """

    phis = rnd.uniform(0, 2*np.pi, (num_samples, dim))
    phases = np.exp(1j*phis)
    return phases
