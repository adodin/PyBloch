from unittest import TestCase
from PyBloch.sampling import sample_magnitudes
import numpy.linalg as la


class TestSample_magnitudes(TestCase):
    def test_sample_magnitudes_shape_1_2(self):
        mags = sample_magnitudes(1, 2)
        self.assertEqual(mags.shape, (1, 2))

    def test_sample_magnitudes_shape_10_2(self):
        mags = sample_magnitudes(10, 2)
        self.assertEqual(mags.shape, (10, 2))

    def test_sample_magnitudes_shape_10_24(self):
        mags = sample_magnitudes(10, 24)
        self.assertEqual(mags.shape, (10, 24))

    def test_sample_magnitudes_array_size(self):
        mags = sample_magnitudes(10, 24)
        self.assertEqual(len(mags[0]), 24)

    def test_sample_magnitudes_norm_1_2(self):
        mags = sample_magnitudes(1, 2)
        self.assertAlmostEqual(la.norm(mags[0]), 1.0)

    def test_sample_magnitudes_norm_10_24(self):
        mags = sample_magnitudes(10, 24)
        self.assertAlmostEqual(la.norm(mags[0]), 1.0)
