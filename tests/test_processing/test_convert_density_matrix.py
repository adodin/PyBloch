from unittest import TestCase

from PyBloch.sampling import sample_psis
from PyBloch.processing import convert_density_matrix
import numpy as np


class TestConvert_density_matrix(TestCase):
    def test_convert_density_matrix_shape_10_2(self):
        psi = sample_psis(10, 2)
        rho = convert_density_matrix(psi)
        self.assertEqual(rho.shape, (10, 2, 2))

    def test_convert_density_matrix_shape_367_2(self):
        psi = sample_psis(367, 2)
        rho = convert_density_matrix(psi)
        self.assertEqual(rho.shape, (367, 2, 2))

    def test_convert_density_matrix_shape_10_24(self):
        psi = sample_psis(10, 24)
        rho = convert_density_matrix(psi)
        self.assertEqual(rho.shape, (10, 24, 24))

    def test_convert_density_matrix_shape_367_24(self):
        psi = sample_psis(367, 24)
        rho = convert_density_matrix(psi)
        self.assertEqual(rho.shape, (367, 24, 24))

    def test_convert_density_matrix_norm_10_2(self):
        psi = sample_psis(10, 2)
        rho = convert_density_matrix(psi)
        norms = rho.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_convert_density_matrix_norm_10_24(self):
        psi = sample_psis(10, 24)
        rho = convert_density_matrix(psi)
        norms = rho.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_convert_density_matrix_norm_367_24(self):
        psi = sample_psis(367, 24)
        rho = convert_density_matrix(psi)
        norms = rho.trace(axis1=-2, axis2=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_convert_density_matrix_hermitian_10_2(self):
        psi = sample_psis(10, 2)
        rho = convert_density_matrix(psi)
        self.assertTrue(np.allclose(np.conj(rho), rho.transpose((0, 2, 1))))

    def test_convert_density_matrix_hermitian_10_24(self):
        psi = sample_psis(10, 24)
        rho = convert_density_matrix(psi)
        self.assertTrue(np.allclose(np.conj(rho), rho.transpose((0, 2, 1))))

    def test_convert_density_matrix_hermitian_367_24(self):
        psi = sample_psis(267, 24)
        rho = convert_density_matrix(psi)
        self.assertTrue(np.allclose(np.conj(rho), rho.transpose((0, 2, 1))))
