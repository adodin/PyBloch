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
    if num_qubits == 1:
        sigmas = proc.convert_density_matrix(psis)
    else:
        sigmas = proc.partial_trace_psi(psis, dims, sys_dims=[0])
    return proc.convert_bloch(sigmas)


def batch_sample_nqubit(num_batch, samples_per_batch, num_qubits):
    xx, yy, zz = [], [], []
    for i in range(num_batch):
        x, y, z = sample_nqubit(samples_per_batch, num_qubits)
        xx.append(x)
        yy.append(y)
        zz.append(z)
    return np.array(xx).flatten(), np.array(yy).flatten(), np.array(zz).flatten()
