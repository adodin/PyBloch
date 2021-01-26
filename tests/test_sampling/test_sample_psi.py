from unittest import TestCase

from PyBloch.sampling import sample_psis
import numpy.linalg as la


class TestSample_psis(TestCase):
    def test_sample_psis_shape_1_2(self):
        psis = sample_psis(1, 2)
        self.assertEqual(psis.shape, (1, 2))

    def test_sample_psis_shape_10_2(self):
        psis = sample_psis(10, 2)
        self.assertEqual(psis.shape, (10, 2))

    def test_sample_psis_shape_10_24(self):
        psis = sample_psis(10, 24)
        self.assertEqual(psis.shape, (10, 24))

    def test_sample_psis_shape_10_24_7(self):
        psis = sample_psis(10, (24, 7))
        self.assertEqual(psis.shape, (10, 24*7))

    def test_sample_psis_array_size(self):
        psis = sample_psis(10, 24)
        self.assertEqual(len(psis[0]), 24)

    def test_sample_psis_norm_1_2(self):
        psis = sample_psis(1, 2)
        self.assertAlmostEqual(la.norm(psis[0]), 1.0)

    def test_sample_psis_norm_10_24(self):
        psis = sample_psis(10, 24)
        self.assertAlmostEqual(la.norm(psis[0]), 1.0)
