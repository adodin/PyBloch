""" Functions for Propagating Quantum Dynamics
Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np

# Default Butcher tableau using the RK4 method
rk4_a = [[0., 0., 0., 0.],
         [0.5, 0., 0., 0.],
         [0., 0.5, 0., 0.],
         [0., 0., 1., 0.]]
rk4_b = [1./6., 1./3., 1./3., 1./6.]
rk4_c = [0., 0.5, 0.5, 1.]


# Runge-Kutta Integrators
def rk_step(y0, t0, dt, deriv, a=rk4_a, b=rk4_b, c=rk4_c, deriv_kwargs ={}):
    """ Function for Performing an Explicit Runge-Kutta Step

    :param y0: Initial State before the step
    :param t0: Initial time before the step
    :param dt: Time step
    :param deriv: Derivative Function. Can have Boolean attribute 'print_drive' to control output of driving field
    :param a: Runge-Kutta a coefficient Butcher Tableau
    :param b: R-K b coefficient Tableau
    :param c: R-K c coefficient Tableau
    :param deriv_kwargs: kwargs for the deriv function
    :return: y, t, (optional: drive) new state, time and optionally external drive if output by deriv function
    """
    k = []
    if hasattr(deriv, 'print_drive'):
        print_drive = deriv.print_drive
    else:
        print_drive = False
    for ai, ci in zip(a, c):
        ti = t0 + ci*dt
        yi = y0
        for i, ki in enumerate(k):
            yi = dt * ai[i] * ki + yi
        if print_drive:
            d_dt, drive = deriv(yi, ti, **deriv_kwargs)
        else:
            d_dt = deriv(yi, ti, **deriv_kwargs)
        k.append(d_dt)
    for bi, ki in zip(b,k):
        y0 = bi * dt * ki + y0
    t0 += dt
    if print_drive:
        return y0, t0, drive
    else:
        return y0, t0


def rk_prop(y0, t0, dt, tf, deriv, deriv_kwargs={}, rk_kwargs ={}):
    """ Runge-Kutta Propagation Function

    :param y0: Initial State
    :param t0: Initial Time
    :param dt: Time Step
    :param tf: Final Time
    :param deriv: Derivative Function. Optional boolean attribute 'print_drive' controls whether to output drive field.
    :param deriv_kwargs: Derivative function Kwargs
    :param rk_kwargs: Runge-Kutta step kwargs
    :return: ys, ts, (optional: drive) Array of states, times and optionally drives if output from deriv function
    """
    if hasattr(deriv, 'print_drive'):
        print_drive = deriv.print_drive
    else:
        print_drive = False
    ys = [y0]
    ts = [t0]
    if print_drive:
        ds = []
    y = y0
    t = t0
    while t < tf:
        if print_drive:
            y, t, drive = rk_step(y, t, dt, deriv, deriv_kwargs=deriv_kwargs, **rk_kwargs, )
            ds.append(drive)
        else:
            y, t = rk_step(y, t, dt, deriv, deriv_kwargs=deriv_kwargs, **rk_kwargs, )
        ys.append(y)
        ts.append(t)
    if print_drive:
        return np.array(ys), np.array(ts), np.array(ds)
    else:
        return np.array(ys), np.array(ts)


def liouville_deriv(y0, t0,  L):
    return L@y0
