import numpy as np
import matplotlib.pyplot as plt
import PyBloch.processing as proc

# Simulation Details
num_sample = 10000
num_scatter = 2000
mu_r = 0.
std_r = 1.
mu_theta = np.pi/2
std_theta = np.pi/6

# Grid Plot details
num_r = 50
num_theta = 50

# Draw Sample Points
rr = np.abs(np.random.normal(mu_r, std_r, size=num_sample))
theta = np.random.normal(mu_theta, std_theta, size=num_sample)
phi = np.random.uniform(0, 2*np.pi, size=num_sample)
theta = theta + np.pi *(phi>np.pi)

# Convert to Cartesian Coordinates
XX = rr * np.sin(theta)*np.cos(phi)
YY = rr * np.sin(theta)*np.sin(phi)
ZZ = rr * np.cos(theta)

# Generate Grids along phi = 0 azimuth
r_grid = np.linspace(0, 1., num_r)
theta_grid = np.linspace(0, 2*np.pi, num_theta)
phi_grid = np.array([0.])
r_grid, theta_grid, phi_grid = np.meshgrid(r_grid, theta_grid, phi_grid)

# Convert to Cartesian Grids
x_grid = r_grid * np.sin(theta_grid)
y_grid = np.zeros_like(x_grid)
z_grid = r_grid * np.cos(theta_grid)

p_est = proc.gaussian_kde(XX, YY, ZZ)
dist = proc.evaluate_kde(x_grid, y_grid, z_grid, p_est)
dist = dist/np.max(dist)


fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.set_ylim([0., 1.])
ax.pcolormesh(theta_grid[:, :, 0], r_grid[:, :, 0], dist[:, :, 0], vmin=0, vmax=1, shading='gouraud')
ax.scatter(theta[0:num_scatter], rr[0:num_scatter], c='w', alpha=0.25)
plt.show()
