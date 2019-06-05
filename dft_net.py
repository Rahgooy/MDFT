import numpy as np
import torch
from torch import nn as nn

from helpers.distances import hotaling_matrix


class DFT_Net(nn.Module):
    def __init__(self, options):
        super(DFT_Net, self).__init__()
        self.__set_options(options)
        self.__set_C()
        self.__set_S()

    def update_S(self):
        self.__set_S()

    def __set_S(self):
        self.H = (torch.eye(2) * 2 - 1 + self.b) / 2
        self.S = torch.zeros((self.options_count, self.options_count))
        for i in range(self.options_count):
            for j in range(self.options_count):
                self.S[i, j] = self.__S(i, j)

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
        }
        self.D = None
        for key in options:
            if options[key] is not None and key in self.options:
                self.options[key] = options[key]
        for key in self.options:
            self.__setattr__(key, self.options[key])

    def forward(self, w, prev_p):
        CM = self.C @ self.M
        V = CM @ w

        SP = self.S @ prev_p
        return SP + V

    def __S(self, i, j):
        dm = self.M[j] - self.M[i]
        d = dm @ self.H @ dm
        d = d * d

        s = self.φ1 * torch.exp(-self.φ2 * d)
        if i == j:
            return 1 - s
        return -s
