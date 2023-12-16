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
tf = 1e8
x_scale = 'log'

# Coupling Parameters
gamma = 1
delta_e = 0.01
delta_g = 0.01

n = 0.05
gammas = gamma*np.array([[1.0, 1.0], [1.0, 1.0]])
gmax = np.max(gammas)
dt = 1/np.max([gmax, delta_e, delta_g])

if x_scale == 'linear':
    ts = np.linspace(0, tf, 1000)
elif x_scale == 'log':
    ts = np.logspace(np.log10(dt)-5, np.log10(tf), num=100, base=10)

# Alignment Parameters
ps_g1 = [1, 0, 1, 0]
ps_g2 = [1, 0, 1, 0]
ps_e1 = [1, 0, 0, 1]
ps_e2 = [1, 0, 0, 1]
ps_par = [1, 0, 0, 0]
ps_X = [1, 0, 0, 0]
colors = ['r', 'k', 'g', 'b']
labels=['full', 'none', 'V', '$\Lambda$']

plt.rc('font', size=15)
plt.rc('lines', linewidth=3)
fig, ax = plt.subplots(2, 2, sharex=True)

ys_arr = []

for p_g1, p_g2, p_e1, p_e2, p_par, p_X, c in zip(ps_g1, ps_g2, ps_e1, ps_e2, ps_par, ps_X, colors):
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

    ax[1, 0].plot(ts, ys[:, 0], c)
    ax[0, 0].plot(ts, ys[:, 2], c)
    ax[1, 1].plot(ts, ys[:, 4], c)
    ax[0, 1].plot(ts, ys[:, 6], c)

# Set x axis scales
for a in ax.flatten():
    if x_scale =='linear':
        a.set_xlim(0, tf)
    elif x_scale =='log':
        a.set_xlim(dt*1e-5, tf)
    a.set_xscale(x_scale)

# Set x axis labels
ax[1, 0].set_xlabel(r'$\gamma t$')
ax[1, 1].set_xlabel(r'$\gamma t$')

# Set y axis  labels
ax[0, 0].set_ylabel(r'$\rho_{e_i,e_i}$', fontsize=20)
ax[1, 0].set_ylabel(r'$\rho_{g_i,g_i}$', fontsize=20)
ax[0, 1].set_ylabel(r'$\rho_{e_1,e_2}$', fontsize=20)
ax[1, 1].set_ylabel(r'$\rho_{g_1,g_2}$', fontsize=20)

# Set y axis for coherences
ax[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
c_min = np.min(ys[:, 4:])
c_max = np.max(ys[:, 4:])
c_range = c_max - c_min
plt.tight_layout()
plt.show()