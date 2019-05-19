import numpy as np
import torch
from torch import nn as nn

from helpers.distances import hotaling_matrix


class DFT_Net(nn.Module):
    def __init__(self, options):
        super(DFT_Net, self).__init__()
        self.__set_options(options)
        self.__set_C()
        self.__set_M()
        self.__set_S()
        self.__set_w()

    def __set_w(self):
        if self.learn_w:
            self.w = np.ones((self.attr_count, 1)) #np.random.uniform(size=(self.attr_count, 1))
            self.w /= self.w.sum()

    def __set_S(self):
        self.H = torch.from_numpy(hotaling_matrix(self.b, dtype=np.float32))
        self.H.requires_grad = False
        self.φ1 = torch.tensor([self.φ1], requires_grad=False)
        self.φ2 = torch.tensor([self.φ2], requires_grad=False)
        self.S = torch.zeros((self.options_count, self.options_count))
        for i in range(self.options_count):
            for j in range(self.options_count):
                self.S[i, j] = self.__S(i, j)

    def __set_M(self):
        if self.learn_m:
            self.M = torch.rand((self.options_count, self.attr_count), requires_grad=True)
        else:
            self.M = torch.tensor(self.M.tolist(), requires_grad=True)

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
            'φ1': 1,
            'φ2': 1,
            'P0': np.zeros((3, 1)),
            'w': np.ones((2, 1)) / 2,
            'learn_m': True,
            'learn_w': False
        }
        self.D = None
        for key in options:
            if key in self.options:
                self.options[key] = options[key]
        for key in self.options:
            self.__setattr__(key, self.options[key])

    def forward(self, w, prev_p):
        if self.learn_m:
            S = torch.zeros((self.options_count, self.options_count))
            for i in range(self.options_count):
                for j in range(self.options_count):
                    S[i, j] = self.__S(i, j)
        else:
            S = self.S

        CM = self.C @ self.M
        V = CM @ w

        SP = S @ prev_p
        return SP + V

    def __S(self, i, j):
        dm = self.M[j] - self.M[i]
        d = dm @ self.H @ dm
        d = d * d

        s = self.φ1 * torch.exp(-self.φ2 * d)
        if i == j:
            return 1 - s
        return -s
