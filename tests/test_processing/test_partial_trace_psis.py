from unittest import TestCase

from PyBloch.sampling import sample_psis
from PyBloch.processing import convert_density_matrix, partial_trace_psi
import numpy as np

psis_pure = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.5, 0.5, 0.5j, 0.5j]])
rho_0 = np.array([[1, 0], [0, 0]])
rho_1 = np.array([[0, 0], [0, 1]])
rho_ihalf = np.array([[0.5, -0.5j], [0.5j, 0.5]])
rho_rhalf = np.array([[0.5, 0.5], [0.5, 0.5]])

psis_bell = 1/np.sqrt(2) * np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1,0], [0, 1, -1, 0]])
rho_bell = np.array([[0.5, 0], [0, 0.5]])


class TestPartial_trace(TestCase):
    def test_partial_trace_dims_bath_dims_shape_10_2_2(self):
        psis = sample_psis(10, 4)
        sigmas = partial_trace_psi(psis, (2, 2), bath_dims=[0])
        self.assertEqual(sigmas.shape, (10, 2, 2))

    def test_partial_trace_dims_sys_dims_shape_10_2_2(self):
        psis = sample_psis(10, 4)
        sigmas = partial_trace_psi(psis, (2, 2), sys_dims=[0])
        self.assertEqual(sigmas.shape, (10, 2, 2))

    def test_partial_trace_dims_bath_dims_shape_10_2_3(self):
        psis = sample_psis(10, 6)
        sigmas = partial_trace_psi(psis, (2, 3), bath_dims=[0])
        self.assertEqual(sigmas.shape, (10, 3, 3))

    def test_partial_trace_dims_sys_dims_shape_10_2_3(self):
        psis = sample_psis(10, 6)
        sigmas = partial_trace_psi(psis, (2, 3), sys_dims=[0])
        self.assertEqual(sigmas.shape, (10, 2, 2))

    def test_partial_trace_dims_bath_dims_shape_10_3_2(self):
        psis = sample_psis(10, 6)
        sigmas = partial_trace_psi(psis, (3, 2), bath_dims=[0])
        self.assertEqual(sigmas.shape, (10, 2, 2))

    def test_partial_trace_dims_sys_dims_shape_10_3_2(self):
        psis = sample_psis(10, 6)
        sigmas = partial_trace_psi(psis, (3, 2), sys_dims=[0])
        self.assertEqual(sigmas.shape, (10, 3, 3))

    def test_partial_trace_dims_one_bath_dims_shape_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0])
        self.assertEqual(sigmas.shape, (10, 15, 15))

    def test_partial_trace_dims_one_sys_dims_shape_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0])
        self.assertEqual(sigmas.shape, (10, 2, 2))

    def test_partial_trace_dims_two_bath_dims_shape_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0, 2])
        self.assertEqual(sigmas.shape, (10, 3, 3))

    def test_partial_trace_dims_two_sys_dims_shape_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0, 2])
        self.assertEqual(sigmas.shape, (10, 10, 10))

    def test_partial_trace_norm_one_bath_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0])
        norms = sigmas.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_partial_trace_norm_two_bath_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0, 2])
        norms = sigmas.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_partial_trace_norm_one_sys_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0])
        norms = sigmas.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_partial_trace_norm_two_sys_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0, 2])
        norms = sigmas.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_partial_trace_hermitian_one_bath_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0])
        self.assertTrue(np.allclose(np.conj(sigmas), sigmas.transpose((0, 2, 1))))

    def test_partial_trace_hermitian_two_bath_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), bath_dims=[0, 2])
        self.assertTrue(np.allclose(np.conj(sigmas), sigmas.transpose((0, 2, 1))))

    def test_partial_trace_hermitian_one_sys_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0])
        self.assertTrue(np.allclose(np.conj(sigmas), sigmas.transpose((0, 2, 1))))

    def test_partial_trace_hermitian_two_sys_dims_10_2_3_5(self):
        psis = sample_psis(10, 30)
        sigmas = partial_trace_psi(psis, (2, 3, 5), sys_dims=[0, 2])
        self.assertTrue(np.allclose(np.conj(sigmas), sigmas.transpose((0, 2, 1))))

    def test_partial_trace_two_qubit_sys_dims_pure_states(self):
        sigmas_0 = partial_trace_psi(psis_pure, (2, 2), sys_dims=[0])
        sigmas_1 = partial_trace_psi(psis_pure, (2, 2), sys_dims=[1])
        expected_0 = np.array([rho_0, rho_0, rho_1, rho_1, rho_ihalf])
        expected_1 = np.array([rho_0, rho_1, rho_0, rho_1, rho_rhalf])
        self.assertTrue(np.allclose(sigmas_0, expected_0))
        self.assertTrue(np.allclose(sigmas_1, expected_1))

    def test_partial_trace_two_qubit_bath_dims_pure_states(self):
        sigmas_0 = partial_trace_psi(psis_pure, (2, 2), bath_dims=[1])
        sigmas_1 = partial_trace_psi(psis_pure, (2, 2), bath_dims=[0])
        expected_0 = np.array([rho_0, rho_0, rho_1, rho_1, rho_ihalf])
        expected_1 = np.array([rho_0, rho_1, rho_0, rho_1, rho_rhalf])
        self.assertTrue(np.allclose(sigmas_0, expected_0))
        self.assertTrue(np.allclose(sigmas_1, expected_1))

    def test_partial_trace_two_qubit_sys_dims_bell_states(self):
        sigmas_0 = partial_trace_psi(psis_bell, (2, 2), sys_dims=[0])
        sigmas_1 = partial_trace_psi(psis_bell, (2, 2), sys_dims=[1])
        expected = np.array([rho_bell, rho_bell, rho_bell, rho_bell])
        self.assertTrue(np.allclose(sigmas_0, expected))
        self.assertTrue(np.allclose(sigmas_1, expected))

    def test_partial_trace_two_qubit_bath_dims_bell_states(self):
        sigmas_0 = partial_trace_psi(psis_bell, (2, 2), bath_dims=[1])
        sigmas_1 = partial_trace_psi(psis_bell, (2, 2), bath_dims=[0])
        expected = np.array([rho_bell, rho_bell, rho_bell, rho_bell])
        self.assertTrue(np.allclose(sigmas_0, expected))
        self.assertTrue(np.allclose(sigmas_1, expected))





