import numpy as np
import torch
from torch import nn as nn


class MDFT_Net(nn.Module):
    def __init__(self, options):
        super(MDFT_Net, self).__init__()
        self.__set_options(options)
        self.__set_C()
        B = self.__invB()
        A = torch.eye(self.attr_count)
        A[-1, -1] = self.b
        self.H = B @ A @ B.T
        self.__set_S()

    def update_S(self):
        self.__set_S()

    def __invB(self):
        n = self.attr_count
        ib = -torch.ones((n, n))
        for i in range(1, n):
            ib[i-1, i] = n - 1
        ib[-1, :] = 1
        return ib / np.sqrt(n)

    def __set_S(self):
        self.S = torch.zeros((self.options_count, self.options_count))
        for i in range(self.options_count):
            for j in range(self.options_count):
                self.S[i, j] = self.__S(i, j)

    def __S(self, i, j):
        dm = self.M[j] - self.M[i]
        d = dm @ self.H @ dm.T
        d = d ** 2

        s = self.phi2 * torch.exp(-self.phi1 * d)
        if i == j:
            return 1 - s
        return -s

    def __set_C(self):
        self.C = torch.ones((self.options_count, self.options_count)) * -(1 / (self.options_count - 1))
        self.C.requires_grad = False
        for i in range(self.options_count):
            self.C[i][i] = 1

    def __set_options(self, options):
        self.options = {
            'b': 10,
            'options_count': 3,
            'attr_count': 2,
            'M': None,
            'phi1': 1,
            'phi2': 1,
            'P0': np.zeros((3, 1)),
            'w': np.ones((2, 1)) / 2,
            'sig2': 1,
            'threshold': 5
        }
        for key in options:
            if options[key] is not None and key in self.options:
                self.options[key] = options[key]
        for key in self.options:
            self.__setattr__(key, self.options[key])

    def forward(self, w, prev_p):
        CM = self.C @ self.M
        E = self.sig2 * self.C @ torch.randn(self.options_count, prev_p.shape[1])
        V = CM @ w
        SP = self.S @ prev_p
        return SP + V + E
