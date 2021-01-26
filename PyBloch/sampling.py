""" Functions for Sampling Initial Conditions

Written By: Amro Dodin (Willard Group - MIYT)
"""

import numpy.random as rnd
import numpy.linalg as la


def sample_psis(num_samples, dim):
    """ Samples Wavefunction Uniformly on Hilbert Space

    :param num_samples: Number of Wavefunctions to Sample
    :param dim: Hilbert Space Dimension
    :return: (num_samples, dim) array of wavefunctions
    """
    # Sample and Normalize Real & Imag components from Gaussian
    comps = rnd.standard_normal((num_samples, 2*dim))
    comps = comps / la.norm(comps, axis=1, keepdims=True)
    psis = comps[:, :dim] + 1j*comps[:, dim:]
    return psis
