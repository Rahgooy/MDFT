import unittest
import numpy as np

from dft import load_DFT_dataset, generate_fixed_time_DFT_samples, get_fixed_T_dft_dist
from helpers.distances import hotaling
import math


class distances_tests(unittest.TestCase):

    def setUp(self):
        self.data = load_DFT_dataset('../data/random/set_2/set_hotaling_n100_l100_1.pickle')

    def test_get_dft_dist(self):
        samples = 100000
        T = 100
        model = self.data
        for i in range(10):
            data = generate_fixed_time_DFT_samples(model.M, model.S, model.w, model.P0, samples, T, {})
            data_samples = np.array([d.choice for d in data.samples])
            data_dist = np.average(data_samples, axis=0)

            dist = get_fixed_T_dft_dist(model, samples, T)

            self.assertTrue(np.allclose(data_dist, dist, rtol=1e-2, atol=1e-3))


if __name__ == "__main__":
    unittest.main()

