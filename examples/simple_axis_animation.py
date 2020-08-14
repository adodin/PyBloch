import numpy as np
import PyBloch.processing as proc
import PyBloch.animation as ani
import scipy.signal as sig

# Generate Data
t = np.linspace(0, 2*np.pi, 5000)
x_init = np.linspace(0, 1, 10)
t, xi = np.meshgrid(t, x_init)

x = xi*np.cos(xi*t)*np.cos(2**xi*t)
y = xi*np.sin(xi*t)*np.cos(2**xi*t)
z = xi*np.sin(2**xi*t)

u, v, w = proc.calculate_axes(x, y, z)
ani.axis_animate(u, v, w,  duration=20, fps=20,
                  fig_kwargs={'figure': "A Simple Acis Animation", 'size': (500, 500), 'bgcolor':(1., 1., 1.), 'fgcolor':(0, 0, 0)},
                  fname='simple_axis_animation.mp4')
