""" Useful Functions for Processing Bloch Sphere Data

Written By: Amro Dodin (Willard Group - MIT)
"""

import numpy as np


def convert_density_matrix(psi):
    """ Convert Wavefunction to Density Matrix

    :param psi: Wavefunction as normalized N-dim complex vector
    :return: (NxN) pure state wavefunction
    """
    return np.matmul(np.expand_dims(psi, -1), np.conj(np.expand_dims(psi, -2)))


def partial_trace_psi(psi, dims, bath_dims=None, sys_dims=None):
    """ Computes reduced system density matrices from total system wavefunction
    This method does not require the storage of total system density matrix.

    :param psi: Array of total system wavefunctions
    :param dims: array of dimensions of each system
    :param bath_dims: indexes of bath hilbert spaces in dimension array
    :param sys_dims:  indexes of system Hilbert spaces in dimension array. NB: use only one of bath_dims or sys_dims
    :return: partially traced density matrix array
    """
    # Check Input Dimensions
    dims = np.array(dims)
    assert len(psi.shape) == 2
    n_sample, d =psi.shape
    assert d == dims.prod()
    n_sys = len(dims)

    # Check that only one array specification is provided & that it is valid
    assert ((bath_dims is None) or (sys_dims is None))
    if bath_dims is not None:
        bath_dims = np.array(bath_dims)
        assert n_sys > len(bath_dims)
        assert n_sys > np.max(bath_dims)
    if sys_dims is not None:
        sys_dims = np.array(sys_dims)
        assert n_sys > len(sys_dims)
        assert n_sys > np.max(sys_dims)

    # Reshape Array Using System Dimensions
    new_shape = np.concatenate([np.array([n_sample]), dims])
    psi_shaped = psi.reshape(new_shape)

    # If bath_dims specified. Find sys_dims
    if bath_dims is not None:
        sys_dims = np.delete(np.arange(n_sys), bath_dims)

    # Construct the Einsum specification
    ein1 = np.concatenate([np.array([Ellipsis]), np.arange(n_sys)])
    ein2 = np.concatenate([np.array([Ellipsis]), np.arange(n_sys)])
    ein2[1 + sys_dims] = ein2[1 + sys_dims] + n_sys
    dim_sigma = dims[sys_dims].prod()
    sigma = np.einsum(psi_shaped, ein1, np.conj(psi_shaped), ein2)
    return sigma.reshape(n_sample, dim_sigma, dim_sigma)


def partial_trace_rho(rho, dims, bath_dims=None, sys_dims=None):
    """ Performs Partial Trace of array of density matrices

    :param rho: Array of density matrices
    :param dims: array of dimensions of each system
    :param bath_dims: indexes of bath hilbert spaces in dimension array
    :param sys_dims:  indexes of system Hilbert spaces in dimension array. NB: use only one of bath_dims or sys_dims
    :return: partially traced over array
    """
    # Grab and Check Density Matrix Dimensions
    dims = np.array(dims)
    assert len(rho.shape) == 3
    n_sample, d1, d2 = rho.shape
    assert d1 == d2
    assert d1 == np.prod(dims)
    n_sys = len(dims)

    # Check that only one array specification is provided & that it is valid
    assert ((bath_dims is None) or (sys_dims is None))
    if bath_dims is not None:
        bath_dims = np.array(bath_dims)
        assert n_sys > len(bath_dims)
        assert n_sys > np.max(bath_dims)
    if sys_dims is not None:
        sys_dims = np.array(sys_dims)
        assert n_sys > len(sys_dims)
        assert n_sys > np.max(sys_dims)

    # Reshape Array Using System Dimensions
    new_shape = np.concatenate([np.array([n_sample]), np.array(dims), np.array(dims)])
    sigma = rho.reshape(new_shape)

    # If sys_dims specified. Find bath_dims
    if sys_dims is not None:
        bath_dims = np.delete(np.arange(n_sys), sys_dims)

    # For Specified Bath Indexes
    assert bath_dims is not None
    bath_dims=np.sort(bath_dims)[::-1]
    for i in bath_dims:
        ax1 = i + 1            # +1 because of run_num occupying index 0
        ax2 = n_sys + i + 1
        sigma = sigma.trace(axis1=ax1, axis2=ax2)
        n_sys = n_sys - 1
    dim_sigma = int(d1/(dims[bath_dims].prod()))

    return sigma.reshape(n_sample, dim_sigma, dim_sigma)


def convert_bloch(rhos):
    """ Convert Density Matrices to cartesian Bloch Sphere coorsinates

    :param rhos: (num_samples, 2, 2) array of sampled qubit density matrrices
    :return: x, y, z real cartesian Bloch Coordinates
    """
    # Check Shape
    n_samples, d1, d2 = rhos.shape
    assert d1 == 2 == d2

    # Calculate Bloch Components
    r00 = rhos[:,0, 0]
    r01 = rhos[:,0, 1]
    r11 = rhos[:,1, 1]
    x = 2*np.real(r01)
    y = 2*np.imag(r01)
    z = np.real(r11-r00)
    return x, y, z


def select_trajectories(x, y, z, n_plot):
    """ Selects a valid set of trajectories. Randomly chosen if n_plot is int.

    :param x ,y, z: (n_traj, n_time) array of Cartesian Bloch spehere coordinates
    :param n_plot: int or [int] how many (randomly chosen) or which trajectories to plot
    :return: x_plot, y_plot, z_plot n_plot trajectories
    """
    # Grab and validate trajectory indexes
    n_traj, n_time = x.shape
    assert x.shape == y.shape == z.shape
    if type(n_plot) is int:
        assert n_plot <= n_traj
        n_plot = np.random.choice(n_traj, n_plot, replace=False)
    else:
        assert max(n_plot) < n_traj

    # Select and return trajectories
    x_plot = np.array([x[i] for i in n_plot])
    y_plot = np.array([y[i] for i in n_plot])
    z_plot = np.array([z[i] for i in n_plot])

    return x_plot, y_plot, z_plot


def calculate_axes(x, y, z):
    """ Computes the instantaneous axes of rotation (as euler vectors) neglecting dephasing and dissipation

    :param x, y, z: (n_traj, n_time) array of Cartesian Bloch coordinates to be plotted (see select_trajectories)
    :return: (3, n_traj, n_time) array of Euler vector rotation axes. NB can be unpacked into u,v,w coordinates
    """

    # Combine coordinate into one array and compute derivative
    r = np.array([x, y, z])
    dr = np.gradient(r, axis=2)

    # Compute Cross Product to get axis of rotation and normalize
    rot = np.cross(r, dr, axis=0)
    norm = np.sqrt(np.sum(rot * rot, axis=0))
    rot = rot/norm

    # Calculate angle of rotation to linear order (small angle approximation
    theta = np.sum(r * dr,  axis=0)/np.sum(r * r, axis=0)

    return rot * theta


def norm_correlate(u, v, w, t_0=0):
    assert u.shape == v.shape == w.shape
    n_traj, n_time = u.shape

    rad = np.sqrt(u**2 + v**2 + w**2)
    rad_0 = rad[:, t_0]
    mu_0 = np.mean(rad_0)
    corr = []

    for i in range(n_time-t_0):
        rad_t = rad[:, t_0+i]
        mu_t = np.mean(rad_t)
        sig_0t = np.mean(rad_0*rad_t)
        corr.append(sig_0t - mu_t*mu_0)

    return np.array(corr)


def anisotropy_correlate(u, v, w, t_0=0):
    assert u.shape == v.shape == w.shape
    n_traj, n_time = u.shape

    r_0 = np.array([u[:, t_0], v[:, t_0], w[:, t_0]])
    mu_0 = np.mean(r_0, axis=-1)
    corr = []

    for i in range(n_time-t_0):
        r_t = np.array([u[:, t_0+i], v[:, t_0+i], w[:, t_0+i]])
        mu_t = np.mean(r_t, axis=-1)
        mu_0t = np.sum(mu_0 * mu_t)
        sig_0t = np.mean(np.sum(r_0 * r_t, axis =0))
        corr.append(sig_0t - mu_0t)

    return np.array(corr)


def gaussian_kde(x, y, z, h=None):
    """ Generate A Gaussian KDE function using a set of samples.

    :param x, y, z: (n_sample) length set of cartesian coordinate samples
    :param h: KDE bandwidth. Use Scott's rule of thumb if None (default: None)
    :return: a function containing the KDE to be called on an array or Meshgrid of points
    """
    assert len(x) == len(y) ==len(z)
    n_sample = len(x)

    # Scott's Rule Bandwidth Estimation
    if h is None:
        h = n_sample**(-1/7)

    # Define Kernel Density Estimator
    def kde(X, Y, Z):
        assert X.shape ==Y.shape == Z.shape
        density = np.zeros_like(X)
        for xx, yy, zz in zip(x, y, z):
            density += np.exp(-((X-xx)**2 + (Y-yy)**2 + (Z-zz)**2)/h**2)
        density = density/np.sum(density)
        return density
    kde.h = h

    return kde


def evaluate_kde(x, y, z, p_est):
    """ Evaluates p_est on a grid defined by the arrays, x, y, z. Also Works with Scipy Estiamtes

    :param x, y, z: 1D Arrays or Mesh Grids defining grid points
    :param p_est:
    :return: distribution evaluated on grid if x,y,z are Mesh grids. Otherwise return X_grid, y_grid,z_grid , dist
    """
    assert x.shape == y.shape == z.shape
    # Checks if points are an array or a meshgrid
    is_array = len(x.shape) == 1

    # If points are arrays, convert to grids
    if is_array:
        x, y, z = np.meshgrid(x, y, z)

    # Flatten Mesh Grids into point list, evaluate and reshape into grids
    pts = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    dist = p_est(pts)
    dist = np.reshape(dist, x.shape)

    if is_array:
        return x, y, z, dist
    else:
        return dist
