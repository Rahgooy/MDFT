import unittest
import numpy as np

from dft import DFT, get_threshold_based_dft_dist
from helpers.distances import hotaling_D, hotaling_S
import math


class SimulatingTests(unittest.TestCase):
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
        S2 = self.distfunct(M, b, φ1, φ2)
        self.assertTrue(np.allclose(S1, S2))

    def test_sim(self):
        M, φ1, φ2, b, σ2, threshold, w = self.M, self.φ1, self.φ2, self.b, self.σ2, self.threshold, self.w
        S1 = hotaling_S(M, φ1, φ2, b)
        S2 = self.distfunct(M, b, φ1, φ2)
        P0 = np.zeros((M.shape[0], 1))
        m = DFT(M, S1, w, P0, σ2)
        f1, T = self.simMDF(S2, m.C, M, w, threshold, σ2, 1000000)
        f2, converged = get_threshold_based_dft_dist(m, 1000000, threshold, False)
        f2 = f2.squeeze()

        self.assertTrue(np.allclose(f1, f2, rtol=0.001, atol=0.001))

    def distfunct(self, M, b, phi1, phi2):
        T = np.array([
            [-1, 1],
            [1, 1]]) / np.sqrt(2)
        n = M.shape[0]
        W = np.diag([1, b])
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                DV = (M[i, :] - M[j, :]).T
                DV = T @ DV
                D[i, j] = DV.T @ W @ DV

        D = - phi2 * np.exp(-phi1 * D ** 2)

        return D

    def simMDF(self, D, C, M, w, threshold, sig2, Ns):
        nd = M.shape[0]
        V3 = C @ M @ w
        P3 = np.zeros(nd)
        T = 0

        P = np.zeros((nd, Ns))
        B = np.zeros(Ns)
        n = Ns
        while (B < threshold).any():
            W = np.random.rand(1, n) < w[0]
            W = np.vstack((W, 1 - W))
            E3 = C @ M @ (W - w) + sig2 * C @ np.random.randn(nd, n)  # compute noise
            P = P + D @ P + V3 + E3  # accumulate
            B = P.max(axis=0)
            converged = B >= threshold
            if converged.any():
                Ind = P[:, converged].argmax(axis=0)
                P3 = P3 + np.array([(Ind == 0).sum(), (Ind == 1).sum(), (Ind == 2).sum()])
            T = T + n
            P = P[:, B < threshold]
            n = P.shape[1]

        P3 = P3 / Ns
        T = T / Ns

        return P3, T
