""" Functions for Sampling Initial Conditions

Written By: Amro Dodin (Willard Group - MIYT)
"""

import numpy.random as rnd
import numpy.linalg as la
import numpy as np
import PyBloch.processing as proc


def sample_psis(num_samples, dim):
    """ Samples Wavefunction Uniformly on Hilbert Space

    :param num_samples: Number of Wavefunctions to Sample
    :param dim: Hilbert Space Dimension or array_like of Hilbert Space Dimensions for Tensor Product
    :return: (num_samples, dim) array of wavefunctions
    """
    # Sample and Normalize Real & Imag components from Gaussian
    dim = np.prod(dim)
    comps = rnd.standard_normal((num_samples, 2*dim))
    comps = comps / la.norm(comps, axis=1, keepdims=True)
    psis = comps[:, :dim] + 1j*comps[:, dim:]
    return psis


def sample_nqubit(num_samples, num_qubits):
    dim = 2 ** num_qubits
    dims = 2*np.ones(num_qubits, dtype=int)
    psis = sample_psis(num_samples, dim)
    rhos = proc.convert_density_matrix(psis)
    if num_qubits == 1:
        sigmas = rhos
        del rhos
    else:
        sigmas = proc.partial_trace(rhos, dims, sys_dims=[0])
        del rhos
    return proc.convert_bloch(sigmas)
