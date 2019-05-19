import numpy as np
from helpers.weight_generator import RouletteWheelGenerator
from pathlib import Path
import pickle


class DFT:
    def __init__(self, M, S, w, P0):
        self.M = M
        self.S = S
        self.w = w
        self.P0 = P0
        self.P = [P0]
        self.V = [np.zeros(P0.shape)]
        self.W = [np.zeros(w.shape)]
        self.t = 0
        self.C = np.ones((M.shape[0], M.shape[0])) * -(1 / (M.shape[0] - 1))
        for i in range(len(M)):
            self.C[i][i] = 1

        self.CM = self.C @ M

    def step(self):
        W = RouletteWheelGenerator(self.w.reshape(-1, 1)).generate()
        V = self.CM @ W
        self.W.append(W)
        self.V.append(V)
        P = self.S @ self.P[self.t] + V
        self.P.append(P)
        self.t += 1

    def get_last_P(self):
        return self.P[self.t]

    def get_last_V(self):
        return self.V[self.t]

    def get_last_W(self):
        return self.W[self.t]


class DFTDataSet:
    def __init__(self, M, S, w, P0, samples, parameters={}):
        self.M = M
        self.S = S
        self.w = w
        self.P0 = P0
        self.C = np.ones((M.shape[0], M.shape[0])) * -(1 / (M.shape[0] - 1))
        for i in range(len(M)):
            self.C[i][i] = 1
        self.samples = samples
        self.parameters = parameters

    def summary(self, file=None):
        steps = np.array([x.t for x in self.samples])
        avg = np.array([d.choice for d in self.samples])
        avg = np.average(avg, axis=0)
        print("========================== Data summary =============================", file=file)
        print("# of examples: {}".format(len(self.samples)), file=file)
        print("max T: {}, min T: {}".format(steps.max(), steps.min()), file=file)
        print("# of options:{}".format(self.M.shape[0]), file=file)
        print("# of attributes:{}".format(self.M.shape[1]), file=file)
        print("W distribution:{}".format(self.w.T), file=file)
        print("M:", file=file)
        print(self.M, file=file)
        print("C:", file=file)
        print(self.C, file=file)
        print("S:", file=file)
        print(self.S, file=file)
        print("Mean option choice: {}".format(avg.T), file=file)
        for key in self.parameters:
            print("{}: {}".format(key, self.parameters[key]), file=file)
        print("=====================================================================", file=file)


def save_DFT_dataset(dataset, path):
    p = Path(path)
    with p.open(mode='wb') as f:
        pickle.dump(dataset, f)


def load_DFT_dataset(path):
    p = Path(path)
    with p.open(mode='rb') as f:
        return pickle.load(f)


class DFTSample:
    def __init__(self, choices, option_idx):
        self.choice = choices[-1]['c']
        self.t = choices[-1]['t']
        self.choices = choices
        self.option_idx = option_idx


def get_dft_dist(model, samples, T):
    def forward(w, prev_p, model):
        CM = model.C @ model.M
        V = CM @ w
        SP = model.S @ prev_p
        return SP + V

    gen = RouletteWheelGenerator(model.w)
    P = np.repeat(model.P0, samples, axis=1)
    for t in range(1, T + 1):
        W = np.array([gen.generate() for _ in range(samples)]).squeeze().T
        P = forward(W, P, model)

    choice_indices = P.argmax(axis=0)
    dist = np.bincount(choice_indices) / samples
    opts = model.M.shape[0]
    if len(dist) < opts:
        dist = np.append(dist, np.zeros(opts - len(dist)))

    return dist.reshape(-1, 1)


def generate_fixed_time_DFT_samples(M, S, w, P0, n, t, parameters):
    samples = []
    for i in range(n):
        dft = DFT(M, S, w, P0)
        choices = [{
            'W': None,
            'P': P0.copy(),
            'V': None,
            'c': None,
            't': 0
        }]
        for j in range(1, t):
            dft.step()
            P = dft.get_last_P()
            V = dft.get_last_V()
            W = dft.get_last_W()
            choice_index = np.argmax(P, axis=0)
            choice = np.zeros(P0.shape)
            choice[choice_index] = 1
            choices.append({
                'W': W,
                'P': P,
                'V': V,
                'c': choice,
                't': j
            })
        samples.append(DFTSample(choices, np.arange(M.shape[0])))
    return DFTDataSet(M, S, w, P0, samples, parameters)


def generate_threshold_based_DFT_samples(M, S, w, P0, n, threshold, parameters):
    samples = []
    for i in range(n):
        dft = DFT(M, S, w, P0)
        choices = [{
            'W': None,
            'P': P0.copy(),
            'V': None,
            'c': None,
            't': 0
        }]
        j = 1
        max_p = 0
        while max_p < threshold:
            dft.step()
            P = dft.get_last_P()
            V = dft.get_last_V()
            W = dft.get_last_W()
            choice_index = np.argmax(P, axis=0)
            choice = np.zeros(P0.shape)
            choice[choice_index] = 1
            choices.append({
                'W': W,
                'P': P,
                'V': V,
                'c': choice,
                't': j
            })
            j += 1
            max_p = P[choice_index] / P.clip(0).sum()
        samples.append(DFTSample(choices, np.arange(M.shape[0])))
    return DFTDataSet(M, S, w, P0, samples, parameters)
