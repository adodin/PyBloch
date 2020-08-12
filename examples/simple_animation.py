import sys
sys.path.append("/Users/amr/Documents/Research/PyBloch")
import numpy as np
import PyBloch.animation as ani
from mayavi import mlab


# Generate Data
t = np.linspace(0, 2*np.pi, 500)
x_init = np.linspace(0, 1, 10)
t, xi = np.meshgrid(t, x_init)

x = xi*np.sin(xi*t)*np.cos(xi*t)
y = xi*np.cos(xi*t)*np.cos(xi*t)
z = xi*np.sin(xi*t)

mlab.options.offscreen=True
ani.bloch_animate(x, y, z, plot_traj=True,
                  fig_kwargs={'figure': "A Simple Animation", 'size': (500, 500), 'bgcolor':(1., 1., 1.), 'fgcolor':(0, 0, 0)},
                  scatter_kwargs={'scale_mode':'none', 'colormap': 'gist_heat', 'resolution':32,'scale_factor': 0.1},
                  line_kwargs={'colormap':'gist_heat', 'tube_radius': 0.01, 'tube_sides':12},
                  fname='outputs/simple_Animation.gif')
