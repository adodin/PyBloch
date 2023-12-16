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
tf = 100
tf_e = 5

# Coupling Parameters
gamma = 1
delta_es = np.flip(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0]))
delta_gs = 0.3*np.ones_like(delta_es)


n = 0.05
gammas = gamma*np.array([[1.5, 1.5], [0.5, 0.5]])

# Detect Suitable Time step
gmax = np.max(gammas)
dt = 1/np.max([gmax, np.max(delta_es), np.max(delta_gs)])

ts = np.linspace(0, tf, 1000)
#ts = np.logspace(np.log10(dt)-3, np.log10(tf), num=100, base=10)

# Alignment Parameters
p_g1 = 1
p_g2 = 1
p_e1 = 1
p_e2 = 1
p_par = 1
p_X = 1

# Derived Parameters
rs = n*gammas
R = np.sum(rs)
G = np.sum(gammas)
g_g1 = np.sqrt(gammas[0, 0] * gammas[0, 1])
g_g2 = np.sqrt(gammas[1, 0] * gammas[1, 1])
g_e1 = np.sqrt(gammas[0, 0] * gammas[1, 0])
g_e2 = np.sqrt(gammas[0, 1] * gammas[1, 1])
g_par = np.sqrt(gammas[0, 0] * gammas[0, 1])
g_X = np.sqrt(gammas[1, 0] * gammas[1, 1])

plt.rc('font', size=15)
plt.rc('lines', linewidth=3)
cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=np.min(delta_es), vmax=np.max(delta_es))

plt.figure()
ax_g = plt.subplot()
plt.figure()
ax_e = plt.subplot()
plt.figure()
ax_gpop = plt.subplot()
fig_p, axes_p = plt.subplots(2, len(delta_gs),  sharey='row', sharex=True, figsize=(12, 4))
fig_c, axes_c = plt.subplots(2, len(delta_gs),  sharey='row', sharex=True, figsize=(12, 4))


for delta_g, delta_e, ax_gp, ax_ep, ax_gc, ax_ec in zip(delta_gs, delta_es, axes_p[1], axes_p[0], axes_c[1], axes_c[0]):
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
    ax_g.plot(ts, np.abs(ys[:, 4] + 1j * ys[:, 5]), c=cmap(norm(delta_e)))
    ax_e.plot(ts, np.abs(ys[:, 6] + 1j * ys[:, 7]), c=cmap(norm(delta_e)))
    ax_gpop.plot(ts, ys[:, 0], c=cmap(norm(delta_e)))

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_g, label=r'$\Delta_e$')
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_e,  label=r'$\Delta_e$')
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_gpop, label=r'$\Delta_e$')

for ax in axes_c.flatten():
    ax.set_xlim(0, tf)
for ax in axes_p.flatten():
    ax.set_xlim(0, tf)

for ax in axes_c[1]:
    ax.set_xlabel(r'$\gamma t$')
for ax in axes_p[1]:
    ax.set_xlabel(r'$\gamma t$')

axes_p[0, 0].set_ylabel(r'$\rho_{e_i, e_i}$')
axes_p[1, 0].set_ylabel(r'$\rho_{g_i, g_i}$')
axes_c[0, 0].set_ylabel(r'$\rho_{e_1, e_2}$')
axes_c[1, 0].set_ylabel(r'$\rho_{g_1, g_2}$')

ax_e.set_xlim(0, tf_e)
ax_g.set_xlim(0, tf)
ax_gpop.set_xlim(0, tf)
ax_e.set_xlabel(r'$\gamma t$')
ax_g.set_xlabel(r'$\gamma t$')
ax_gpop.set_xlabel(r'$\gamma t$')
ax_e.set_ylabel(r'$|\rho_{e_1e_2}|$')
ax_g.set_ylabel(r'$|\rho_{g_1g_2}|$')
ax_gpop.set_ylabel(r'$\rho_{g_1g_1}$')
ax_e.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax_g.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax_e.set_ylim(0, 8e-3)
ax_g.set_ylim(0, 6.5e-3)

plt.tight_layout()

plt.show()
