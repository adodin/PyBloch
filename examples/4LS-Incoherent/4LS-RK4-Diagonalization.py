"""
Propagating the PSBR equations for a 4LS in an incoherent light field
Amro Dodin
"""

import numpy as np
import matplotlib.pyplot as plt

import PyBloch.propagation as prop

# Initial State
y0 = np.array([1/2, 1/2, 0, 0, 0, 0, 0, 0])

# Propagation Parameters
tf = 1000
tf_explicit = 10**6

# Coupling Parameters
gamma = 1
delta_g = 0.01
delta_e = 10

n = 0.05
gammas = gamma*np.array([[1, 1], [1, 1]])

# Detect Suitable Time step
gmax = np.max(gammas)
dt = 0.1/np.max([gmax, delta_e, delta_g])

t_plot = np.logspace(np.log10(dt)-2, np.log10(tf_explicit), num=100, base=10)

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

# Labels Array for Legend
labels = [r'$\rho_{g_1,g_1}$', r'$\rho_{g_2,g_2}$', r'$\rho_{e_1, e_1}$', r'$\rho_{e_2, e_2}$',
          r'$\rho_{g_1, g_2}^R$', r'$\rho_{g_1, g_2}^I$', r'$\rho_{e_1, e_2}^R$', r'$\rho_{e_1, e_2}^I$']
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

#  RK propagation
ys, ts = prop.rk_prop(y0, t0=0, tf=tf, dt=dt, deriv=prop.liouville_deriv, deriv_kwargs={'L': L})

# Exact Diagonalization
ys_diag, t_diag = prop.exact_diag(y0, t_plot, L)

plt.rc('font', size=20)
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(ts, ys[:, 0], 'r')
ax[0, 0].plot(ts, ys[:, 1], 'b', alpha=0.5)
ax[0, 0].scatter(t_diag, ys_diag[:, 0], c='r')
ax[0, 0].scatter(t_diag, ys_diag[:, 1], c='b', alpha=0.5)
ax[0, 1].plot(ts, ys[:, 2], 'r')
ax[0, 1].plot(ts, ys[:, 3], 'b', alpha=0.5)
ax[0, 1].scatter(t_diag, ys_diag[:, 2], c='r')
ax[0, 1].scatter(t_diag, ys_diag[:, 3], c='b', alpha=0.5)
ax[1, 0].plot(ts, ys[:, 4], 'r')
ax[1, 0].plot(ts, ys[:, 5], 'b', alpha=0.5)
ax[1, 0].scatter(t_diag, ys_diag[:, 4], c='r')
ax[1, 0].scatter(t_diag, ys_diag[:, 5], c='b', alpha=0.5)
ax[1, 1].plot(ts, ys[:, 6], 'r')
ax[1, 1].plot(ts, ys[:, 7], 'b', alpha=0.5)
ax[1, 1].scatter(t_diag, ys_diag[:, 6], c='r')
ax[1, 1].scatter(t_diag, ys_diag[:, 7], c='b', alpha=0.5)

#plt.plot(ts, ys[:, 4], 'r')
#plt.plot(ts, ys[:, 5], 'b', alpha=0.5)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xlabel(r'$\gamma t$')
#plt.ylabel(r'$\rho_{g_1,g_2}$')
#plt.title(r"$\Delta_g$={}$\gamma$, $\Delta_e$={}$\gamma$".format(delta_g, delta_e),
#          fontdict={'size':20})
#plt.ylim(-0.007,  0.007)
#plt.xlim(0, tf_explicit)

for a in ax.flatten():
    a.set_xscale('log')
ax[1,1].set_xscale('log')
plt.tight_layout()

plt.show()
