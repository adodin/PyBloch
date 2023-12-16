""" Tests Qubit Sampling Algorithm

Written By: Amro Dodin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PyBloch.sampling as smp
from mpl_toolkits.mplot3d import Axes3D

# Simulation Parameters
num_samples = 1000000
num_plotted = 1000
num_bins = 100

# Sample Wave Function
x, y, z = smp.sample_nqubit(num_samples, 1)
r = np.sqrt(x**2 + y**2 + z**2)

# Generate Histogram Bins
cart_bins = np.linspace(-1, 1, num_bins+1)
del_cart = cart_bins[1] - cart_bins[0]
rad_bins = np.linspace(0, 1, num_bins+1)
del_rad = rad_bins[1] - rad_bins[0]

# Set Font Size
font = {'family': 'serif', 'size': 22}
mpl.rc('font', **font)

# Plot Histograms
fig, axs = plt.subplots(2, 2, figsize=[10, 10])
axs[0,0].hist(x, cart_bins, density=True)
axs[0,0].axhline(0.5, c='red', lw=3, ls="--")
axs[0,0].set_xlim(-1, 1)
axs[0,0].set_ylim(0, 0.75)
axs[0,0].set_xlabel('x')

axs[0,1].hist(y, cart_bins, density=True)
axs[0,1].axhline(0.5, c='red', lw=3, ls="--")
axs[0,1].set_xlim(-1, 1)
axs[0,1].set_ylim(0, 0.75)
axs[0,1].set_xlabel('y')

axs[1,0].hist(z, cart_bins, density=True)
axs[1,0].axhline(0.5, c='red', lw=3, ls="--")
axs[1,0].set_xlim(-1, 1)
axs[1,0].set_ylim(0, 0.75)
axs[1,0].set_xlabel('z')

axs[1,1].hist(r, rad_bins, density=True)
axs[1,1].axvline(1-del_rad/2, c='red', lw=3, ls='--')
axs[1,1].set_xlim(0.9, 1.1)
axs[1,1].set_ylim(0, 125)
axs[1,1].set_xlabel('r')

plt.savefig('./outputs/one_qubit_hists.eps')
plt.savefig('./outputs/one_qubit_hists.png')
plt.show()

fig = plt.figure(figsize=[7, 10])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:num_plotted], y[:num_plotted], z[:num_plotted])
plt.savefig('./outputs/one_qubit_bloch.eps')
plt.savefig('./outputs/one_qubit_bloch.png')
plt.show()
