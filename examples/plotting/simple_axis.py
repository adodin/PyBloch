import numpy as np
import PyBloch.processing as proc
import PyBloch.plotting as plt

# Generate Data
t = np.linspace(0, 2*np.pi, 500)
x_init = np.linspace(0, 1, 10)
t, xi = np.meshgrid(t, x_init)

x = xi*np.cos(xi*t)*np.cos(2**xi*t)
y = xi*np.sin(xi*t)*np.cos(2**xi*t)
z = xi*np.sin(2**xi*t)

u, v, w = proc.calculate_axes(x, y, z)

pts = plt.bloch_axis(u, v, w, frame=0,
                     fig_kwargs={'figure': "A Simple Axis Plot", 'size': (1000, 1000),
                                 'bgcolor':(1., 1., 1.), 'fgcolor':(0, 0, 0)},
                     fname='outputs/simple_axis.png', show_fig=True)
