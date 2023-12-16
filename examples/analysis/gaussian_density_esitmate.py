import numpy as np
import matplotlib.pyplot as plt
import PyBloch.processing as proc

# Simulation Details
num_sample = 10000
num_scatter = 2000
grid_resolution = 50
z_level = 25

# Distribution Details
mu_x = 0.
std_x = 0.01
mu_y = 0.5
std_y = 0.5
mu_z = 0.
std_z = 100.

# Sample Points
x = np.random.normal(mu_x, std_x, num_sample)
y = np.random.normal(mu_y, std_y, num_sample)
z = np.random.normal(mu_z, std_z, num_sample)

# Construct Grids
grid_arr = np.linspace(-1, 1, grid_resolution)
XX, YY, ZZ = np.meshgrid(grid_arr, grid_arr, grid_arr)

# Evaluate Density Estimate
print("Evaluating Density Estimate")
kde = proc.gaussian_kde(x, y, z)
print("Computing Density on Grid")
print("Density Bandwidth is: " + str(kde.h))
density = kde(XX, YY, ZZ)
density = density/np.max(density)

tester = np.cos(XX**2 + YY**2 + ZZ**2)
plt.imshow(density[:, :, z_level], interpolation='bilinear',
           origin='lower', extent=[-1., 1., -1., 1.], vmin=0., vmax=1.)
plt.show()
