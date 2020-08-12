import numpy as np
import PyBloch.plotting as plt
from mayavi import mlab

# Generate Data
t = np.linspace(0, 2*np.pi, 500)
x_init = np.linspace(0, 1, 10)
t, xi = np.meshgrid(t, x_init)

x = xi*np.cos(xi*t)*np.cos(2**xi*t)
y = xi*np.sin(xi*t)*np.cos(2**xi*t)
z = xi*np.sin(2**xi*t)

mlab.options.offscreen=True
pts, trajs = plt.bloch_plot(x, y, z, frame=499,
                            fig_kwargs={'figure': "A Simple Plot", 'size': (1000, 1000), 'bgcolor':(1., 1., 1.), 'fgcolor':(0, 0, 0)},
                            scatter_kwargs={'scale_mode':'none', 'colormap': 'gist_heat', 'resolution':32,'scale_factor': 0.1},
                            line_kwargs={'colormap':'gist_heat', 'tube_radius': 0.01, 'tube_sides':12},
                            fname='outputs/simple_plot.png', show_fig=False)
