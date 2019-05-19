import unittest
from dft_net import DFT_Net
from helpers.distances import hotaling, hotaling_S_from_D
import numpy as np
from dft import DFT
import torch


class DFT_net_tests(unittest.TestCase):

    def setUp(self):
        self.model = DFT_Net(3, 2)
        self.model.M[0, 0] = 1
        self.model.M[0, 1] = 3
        self.model.M[1, 0] = 2
        self.model.M[1, 1] = 2
        self.model.M[2, 0] = 3
        self.model.M[2, 1] = 1
        self.M = self.model.M.data.numpy().copy()
        self.model.φ1 = 0.1
        self.model.φ2 = 0.1
        self.D = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.D[i][j] = hotaling(self.M[i], self.M[j], self.model.b)
        self.S = hotaling_S_from_D(self.D, self.model.φ1, self.model.φ2)

    def test_S(self):
        for i in range(3):
            for j in range(3):
                p = self.model.__S(i, j).data.numpy()[0, 0]
                a = self.S[i, j]
                self.assertTrue(np.allclose(a, p), msg="S[{}][{}]".format(i, j))

    def test_step(self):
        for _ in range(20):
            w = [[.45], [.55]]
            p0 = np.zeros((3, 1))
            dft = DFT(self.M, self.S, np.array(w), p0)
            p = torch.Tensor(p0.tolist())
            for t in range(10):
                dft.step()
                a = dft.get_last_P()
                p = self.model.forward(torch.Tensor(dft.W[t + 1].tolist()), p)
                self.assertTrue(np.all(np.abs(a - p.data.numpy()) < 1e-4), msg="step {}, a={}, p={}".format(t + 1, a, p))


if __name__ == "__main__":
    unittest.main()
