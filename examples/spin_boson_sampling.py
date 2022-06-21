"""
Example for thermally sampling Spin-Boson Wavefunction

Written By: Amro Dodin (MIT - Willard Group)
"""

import numpy as np
import numpy.random as rng
import PyBloch.sampling as smp
import matplotlib.pyplot as plt
import PyBloch.processing as proc


def compute_sb_cutoffs(omegas, kT, threshold=0.001):
    """ Computes Thermal Cutoffs for Boson Systems (i.e. how many states needed per oscillator)

    :param omegas: Frequency Array for each System. First Element [0] is the spin frequency
    :param kT: Thermal Energ
    :param threshold: Error Threshold (max probability allowed for truncated states)
    :return: List of Dimensions giving number of states for each system
    """
    dims = [2]
    for w in omegas[1:]:
        dims.append(max(1, int(-kT*np.log(threshold)/w)))
    return dims


def construct_ham(omegas, dims):
    """ Constructs a Hamiltonian for Combination of Spin/Boson Systems

    :param omegas: Frequency Array for each system
    :param dims: Array of Dimensions for each system. States per system
    :return: A Colelctive Hamiltonian in eigenbasis. Array giving diagonal elements.
    """
    hams = []
    for w, d in zip(omegas, dims):
        hams.append(w*np.arange(d))
    ham_T = np.sum(np.meshgrid(*hams, indexing='ij'), axis=0)
    return ham_T.flatten()


def monte_carlo_sample(ham, kT, num_samples, psi_0=None, print_prog=False):
    dim = len(ham)
    if psi_0 is None:
        psi_0 = smp.sample_psis(1, dim)[0]

    samples = []
    num = 0
    E_0 = proc.calculate_energy(psi_0, ham)
    while num < num_samples:
        trial = smp.sample_psis(1, dim)[0]
        E_trial = proc.calculate_energy(trial, ham)
        threshold = np.exp(-(E_trial-E_0)/kT)
        test = rng.sample()
        if test < threshold:
            samples.append(trial)
            psi_0 = trial
            E_0 = E_trial
            num += 1
            if print_prog:
                print(str(num) + '/' + str(num_samples))
    return samples


if __name__ == "__main__":
    # Simulation properties
    kT = 0.15  # Thermal Energy
    omegas = [1., 1.0, 2.0, 3.0, 4.0, 5.]  # Energy scales (0th is spin remaining are oscillators

    dims = compute_cutoffs(omegas, kT)
    H_T = construct_ham(omegas, dims)
    samples = np.array(monte_carlo_sample(H_T, 1000000, print_prog=True))

    rhos = proc.partial_trace_psi(samples, dims, sys_dims=[0])
    sigma = proc.convert_bloch(rhos)
    np.save('outputs/open_sigma-T_0.15w0.npy', sigma)

    bin_edges = np.linspace(-1, 1, 51)
    bins = np.linspace(-1, 1, 50)
    x_h, b = np.histogram(sigma[0], bin_edges, density=True)
    y_h, b = np.histogram(sigma[1], bin_edges, density=True)
    z_h, b = np.histogram(sigma[2], bin_edges, density=True)
    plt.plot(bins, x_h)
    plt.plot(bins, y_h)
    plt.plot(bins, z_h)
    plt.legend(['x', 'y', 'z'])
    plt.xlim(-1, 1)
    plt.show()
