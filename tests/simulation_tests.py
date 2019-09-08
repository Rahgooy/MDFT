import unittest
import numpy as np

from busemeyer.distfunct import distfunct
from busemeyer.simMDF import simMDF
from dft import DFT, get_threshold_based_dft_dist
from helpers.distances import hotaling_D, hotaling_S
import math


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.M = np.array([
            [5.9298, 7.6541],
            [2.1067, 8.4122],
            [8.5224, 5.7808]
        ])
        self.w = np.array([
            [0.3],
            [0.7]
        ])
        self.φ1 = 0.025889
        self.φ2 = 0.021394
        self.b = 9
        self.threshold = 13
        self.σ2 = 0.10152
        pass

    def test_buesmeyer_dist(self):
        M, φ1, φ2, b = self.M, self.φ1, self.φ2, self.b
        S1 = hotaling_S(M, φ1, φ2, b) - np.eye(3)
        S2, _ = distfunct(M, b, φ1, φ2)
        self.assertTrue(np.allclose(S1, S2))

    def test_sim(self):
        M, φ1, φ2, b, σ2, threshold, w = self.M, self.φ1, self.φ2, self.b, self.σ2, self.threshold, self.w
        S1 = hotaling_S(M, φ1, φ2, b)
        S2, _ = distfunct(M, b, φ1, φ2)
        P0 = np.zeros((M.shape[0], 1))
        m = DFT(M, S1, w, P0, σ2)
        f1, T = simMDF(S2, m.C, M, w, threshold, σ2, 1000000)
        f2, converged = get_threshold_based_dft_dist(m, 1000000, threshold, False)
        f2 = f2.squeeze()

        self.assertTrue(np.allclose(f1, f2, rtol=0.001, atol=0.001))
