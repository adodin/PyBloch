from unittest import TestCase

from PyBloch.sampling import sample_phases
import numpy as np

class TestSample_phases(TestCase):
    def test_sample_phases_shape_1_2(self):
        phases = sample_phases(1, 2)
        self.assertEqual(phases.shape, (1, 2))

    def test_sample_phases_shape_10_2(self):
        phases = sample_phases(10, 2)
        self.assertEqual(phases.shape, (10, 2))

    def test_sample_phases_shape_10_24(self):
        phases = sample_phases(10, 24)
        self.assertEqual(phases.shape, (10, 24))

    def test_sample_phases_array_size(self):
        phases = sample_phases(10, 24)
        self.assertEqual(len(phases[0]), 24)

    def test_sample_phases_norm_1_2(self):
        phases = sample_phases(1, 2)
        self.assertTrue(np.allclose(abs(phases), np.ones_like(phases)))

    def test_sample_phases_norm_10_24(self):
        phases = sample_phases(10, 24)
        self.assertTrue(np.allclose(abs(phases), np.ones_like(phases)))
