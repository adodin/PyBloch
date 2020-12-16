""" Utility Functions for external drives
Written by: Amro Dodin
"""

from numpy import cos, exp, sin


def cos_drive(t, delta):
    return cos(delta*t)


def chirp_cos_drive(t, delta_0, delta_f, chirp):
    assert chirp > 0
    t_norm = min(t / chirp, 1)
    omega = delta_f * t_norm + delta_0 * (1 - t_norm)
    return cos(omega * t)


def two_cos_drive(t, delta_0, delta_f, A_f):
    return A_f * cos(delta_f * t) + (1 - A_f) * cos(delta_0 * t)


def two_cos_chirp_drive(t, delta_0, delta_f, chirp):
    assert chirp >0
    t_norm = min(t / chirp, 1)
    return t_norm * cos(delta_f*t) + (1-t_norm)*cos(delta_0 * t)


def am_cos_drive(t, omega, A_0, A_f, chirp):
    assert chirp >0
    t_norm = min(t/chirp, 1)
    return (t_norm * A_f + (1-t_norm) * A_0)*cos(omega*t)


def fixed_amp_cos_drive(t, omega, A):
    return A * cos(omega * t)
