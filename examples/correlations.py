import numpy as np
import PyBloch.processing as proc
import matplotlib.pyplot as plt
import PyBloch.animation as ani

# Simulation Details
num_traj = 2000
num_time = 1000
kick_size = 0.1

# Initialize Systems
u, v, w = [np.random.normal(size=num_traj)], [np.random.normal(size=num_traj)], [np.random.normal(size=num_traj)]

for i in range(num_time):
    u.append(u[-1] + np.random.normal(scale=kick_size, size=num_traj))
    v.append(v[-1] + np.random.normal(scale=kick_size, size=num_traj))
    w.append(w[-1] + np.random.normal(scale=kick_size, size=num_traj))

u = np.array(u).T
v = np.array(v).T
w = np.array(w).T

n_corr = proc.norm_correlate(u, v, w)
vec_cor = proc.anisotropy_correlate(u, v, w)

plt.plot(n_corr, 'r')
plt.show()
plt.plot(vec_cor, 'b')
plt.show()

ani.axis_animate(u, v, w, fname='correlate_test.mp4')
