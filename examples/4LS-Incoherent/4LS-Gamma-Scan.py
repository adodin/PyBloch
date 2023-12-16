"""
Propagating the PSBR equations for a 4LS in an incoherent light field
Amro Dodin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import PyBloch.propagation as prop

# Initial State
y0 = np.array([1/2, 1/2, 0, 0, 0, 0, 0, 0])

# Propagation Parameters
tf = 50
tf_e = 5

# Coupling Parameters
gamma = 1
delta_e = 10
delta_g = 10

n = 0.05
gammas_arr = [gamma*np.array([[1.0, 1.0], [1.0, 1.0]]),
          gamma*np.array([[1.5, 1.5], [0.5, 0.5]]),
          gamma*np.array([[1.5, 0.5], [1.5, 0.5]])]


ts = np.linspace(0, tf, 1000)
#ts = np.logspace(np.log10(dt)-3, np.log10(tf), num=100, base=10)

# Alignment Parameters
p_g1 = 1
p_g2 = 1
p_e1 = 1
p_e2 = 1
p_par = 1
p_X = 1

plt.rc('font', size=15)
plt.rc('lines', linewidth=3)

fig_p, axes_p = plt.subplots(2, len(gammas_arr),  sharey='row', sharex=True, figsize=(12, 4))
fig_c, axes_c = plt.subplots(2, len(gammas_arr),  sharey='row', sharex=True, figsize=(12, 4))

ys_arr = []

for gammas, ax_gp, ax_ep, ax_gc, ax_ec in zip(gammas_arr, axes_p[1], axes_p[0], axes_c[1], axes_c[0]):
    # Derived Parameters
    rs = n * gammas
    R = np.sum(rs)
    G = np.sum(gammas)
    g_g1 = np.sqrt(gammas[0, 0] * gammas[0, 1])
    g_g2 = np.sqrt(gammas[1, 0] * gammas[1, 1])
    g_e1 = np.sqrt(gammas[0, 0] * gammas[1, 0])
    g_e2 = np.sqrt(gammas[0, 1] * gammas[1, 1])
    g_par = np.sqrt(gammas[0, 0] * gammas[0, 1])
    g_X = np.sqrt(gammas[1, 0] * gammas[1, 1])

    """
    We are about to generate a bunch of Liouvillians as matrices in the basis
        [0] - rho_g1,g1: g_1 population
        [1] - rho_g2,g2: g_2 population
        [2] - rho_e1,e1: e_1 population
        [3] - rho_e1,e1: e_2 population
        [4] - rho_g1,g2^R: Real part of ground state coherence
        [5] - rho_g1,g2^I: Imaginary part of ground state coherence
        [6] - rho_e1,e2^R: Real part of excited state coherence
        [7] - rho_e1,e2^I: Imaginary part of excited state coherence
    """

    # -------------------------------------------
    # Generate Liouvillian for Coherent Evolution
    # -------------------------------------------
    L_u = np.zeros((8, 8))
    L_u[4, 5] = delta_g
    L_u[5, 4] = -delta_g
    L_u[6, 7] = delta_e
    L_u[7, 6] = -delta_e

    # -------------------------------------------
    # Secular Liouvillian
    # -------------------------------------------
    L_sec = np.zeros((8, 8))

    # Secular Excitation
    # Population Leaving Ground State b/c Excitation
    L_sec[0, 0] = -(rs[0, 0] + rs[0, 1])
    L_sec[1, 1] = -(rs[1, 0] + rs[1, 1])
    # Populations Entering Excited State b/c Excitation
    # gi -> e1
    L_sec[2, 0] = rs[0, 0]
    L_sec[2, 1] = rs[1, 0]
    # gi -> e2
    L_sec[3, 0] = rs[0, 1]
    L_sec[3, 1] = rs[1, 1]

    # Secular Decay
    # Population Leaving Excited State b/c Spontaneous + Stimulated Emission
    L_sec[2, 2] = -(1+n) * (gammas[0, 0] + gammas[1, 0])
    L_sec[3, 3] = -(1+n) * (gammas[0, 1] + gammas[1, 1])
    # Populations Entering Ground State b/c Emission
    # ei -> g1
    L_sec[0, 2] = (1 + n) * gammas[0, 0]
    L_sec[0, 3] = (1 + n) * gammas[0, 1]
    # ei -> g2
    L_sec[1, 2] = (1 + n) * gammas[1, 0]
    L_sec[1, 3] = (1 + n) * gammas[1, 1]

    # T_1 Coherence decay (due to loss of population)
    # Loss of g1g2 due to excitation
    L_sec[4, 4] = -0.5 * R
    L_sec[5, 5] = -0.5 * R
    # Loss of  e1e2 due to emission
    L_sec[6, 6] = -0.5 * (G + R)
    L_sec[7, 7] = -0.5 * (G + R)


    # ------------------------------------------------------
    # Non-Secular (Interference) Liouvillian
    # ------------------------------------------------------
    L_ns = np.zeros((8, 8))

    # Interference in Excitation
    # Exciting Coherent Superposition
    L_ns[6, 0] = n * p_g1 * g_g1
    L_ns[6, 1] = n * p_g2 * g_g2
    # Faster population transfer
    L_ns[0, 4] = -n * (p_e1 * g_e1 + p_e2 * g_e2)
    L_ns[1, 4] = -n * (p_e1 * g_e1 + p_e2 * g_e2)
    L_ns[2, 4] = 2 * n * p_e1 * g_e1
    L_ns[3, 4] = 2 * n * p_e2 * g_e2
    # Faster GS coherence loss b/c excitation interference
    L_ns[4, 0] = -0.5 * n * (p_e1 * g_e1 + p_e2 * g_e2)
    L_ns[4, 1] = -0.5 * n * (p_e1 * g_e1 + p_e2 * g_e2)

    # Interference in Emission
    # Emitting to Coherent Superposition
    L_ns[4, 2] = (1 + n) * p_e1 * g_e1
    L_ns[4, 3] = (1 + n) * p_e2 * g_e2
    # Faster Population Transfer
    L_ns[2, 6] = -(1 + n) * (p_g1 * g_g1 + p_g2 * g_g2)
    L_ns[3, 6] = -(1 + n) * (p_g1 * g_g1 + p_g2 * g_g2)
    L_ns[0, 6] = 2 * (1 + n) * p_g1 * g_g1
    L_ns[1, 6] = 2 * (1 + n) * p_g2 * g_g2
    # Faster ES coherence loss b/c emission interference
    L_ns[6, 2] = -0.5 * (1 + n) * (p_g1 * g_g1 + p_g2 * g_g2)
    L_ns[6, 3] = -0.5 * (1 + n) * (p_g1 * g_g1 + p_g2 * g_g2)

    # Coherence Transfer
    # GS -> ES Coherence Transfer via Excitation
    L_ns[6, 4] = n * (p_par * g_par + p_X * g_X)
    L_ns[7, 5] = n * (p_par * g_par - p_X * g_X)
    # ES -> GS Coherence Transfer via Emission
    L_ns[4, 6] = (1 + n) * (p_par * g_par + p_X * g_X)
    L_ns[5, 7] = (1 + n) * (p_par * g_par - p_X * g_X)

    # ------------------------------------------------------
    # Propagation
    # ------------------------------------------------------
    L = L_sec + L_u + L_ns

    ys, ts = prop.exact_diag(y0, ts, L)
    ys_arr.append(ys)

    ax_gp.plot(ts, ys[:, 0], 'r')
    ax_gp.plot(ts, ys[:, 1], 'b')
    ax_ep.plot(ts, ys[:, 2], 'r')
    ax_ep.plot(ts, ys[:, 3], 'b')
    ax_gc.plot(ts, ys[:, 4], 'r')
    ax_gc.plot(ts, ys[:, 5], 'b')
    ax_gc.plot(ts, np.abs(ys[:, 4]+1j*ys[:, 5]), 'k')
    ax_ec.plot(ts, ys[:, 6], 'r')
    ax_ec.plot(ts, ys[:, 7], 'b')
    ax_ec.plot(ts, np.abs(ys[:, 6] + 1j * ys[:, 7]), 'k')


for ax in axes_c.flatten():
    ax.set_xlim(0, tf)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
for ax in axes_p.flatten():
    ax.set_xlim(0, tf)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

for ax in axes_c[1]:
    ax.set_xlabel(r'$\gamma t$')
for ax in axes_p[1]:
    ax.set_xlabel(r'$\gamma t$')

axes_p[0, 0].set_ylabel(r'$\rho_{e_i, e_i}$')
axes_p[1, 0].set_ylabel(r'$\rho_{g_i, g_i}$')
axes_c[0, 0].set_ylabel(r'$\rho_{e_1, e_2}$')
axes_c[1, 0].set_ylabel(r'$\rho_{g_1, g_2}$')

fig_pop, axes_pop = plt.subplots(2,  sharex=True, figsize=(8, 8))
h_g_eq = axes_pop[1].plot(ts, ys_arr[0][:, 0], 'r')
h_g_gneq1 = axes_pop[1].plot(ts, ys_arr[1][:, 0], 'g')
h_g_gneq2 = axes_pop[1].plot(ts, ys_arr[1][:, 1], 'g--')
h_g_eneq = axes_pop[1].plot(ts, ys_arr[2][:, 0], 'b')
h_e_eq = axes_pop[0].plot(ts, ys_arr[0][:, 2], 'r')
h_e_gneq = axes_pop[0].plot(ts, ys_arr[1][:, 2], 'g')
h_e_eneq1 = axes_pop[0].plot(ts, ys_arr[2][:, 2], 'b')
h_e_eneq2 = axes_pop[0].plot(ts, ys_arr[2][:, 3], 'b--')

axes_pop[0].legend(labels=[r'$\rho_{e_ie_i}: \gamma_{g_ie_j}=\gamma$',
                           r'$\rho_{e_ie_i}: \gamma_{g_1e_i}=1.5\gamma, \gamma_{g_2e_i}=0.5\gamma$',
                           r'$\rho_{e_1e_1}: \gamma_{g_ie_1}=1.5\gamma, \gamma_{g_ie_2}=0.5\gamma$',
                           r'$\rho_{e_2e_2}: \gamma_{g_ie_1}=1.5\gamma, \gamma_{g_ie_2}=0.5\gamma$'],
                   loc='lower right', fontsize=16)
axes_pop[1].legend(labels=[r'$\rho_{g_ig_i}: \gamma_{g_ie_j}=\gamma$',
                           r'$\rho_{g_1g_1}: \gamma_{g_1e_i}=1.5\gamma, \gamma_{g_2e_i}=0.5\gamma$',
                           r'$\rho_{g_2g_2}: \gamma_{g_1e_i}=1.5\gamma, \gamma_{g_2e_i}=0.5\gamma$',
                           r'$\rho_{g_ig_i}: \gamma_{g_ie_1}=1.5\gamma, \gamma_{g_ie_2}=0.5\gamma$'],
                   loc='upper right', fontsize=16)
axes_pop[1].set_xlim(0, tf)

axes_pop[1].set_xlabel(r'$\gamma t$', fontsize=16)
axes_pop[1].set_ylabel(r'$\rho_{g_ig_i}$',fontsize=16)
axes_pop[0].set_ylabel(r'$\rho_{e_ie_i}$', fontsize=16)

plt.show()
