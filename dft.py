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


def get_fixed_T_dft_dist(model, samples, T):
    dist, _ = get_dft_dist(model, samples, False, T, 0)
    return dist


def get_threshold_based_dft_dist(model, samples, threshold):
    return get_dft_dist(model, samples, True, 0, threshold)


def get_dft_dist(model, samples, tb, T, threshold):
    def forward(w, prev_p, model):
        CM = model.C @ model.M
        V = CM @ w
        SP = model.S @ prev_p
        return SP + V

    gen = RouletteWheelGenerator(model.w)
    P = np.repeat(model.P0, samples, axis=1)
    has_converged = True
    if tb:
        s = samples
        converged = None

        t = 1
        MAX_T = 2000
        while s > 0 and t < MAX_T:
            t += 1
            W = np.array([gen.generate() for _ in range(s)], dtype=np.double).squeeze().T
            if W.ndim == 1:
                W = W.reshape(-1, s)
            P = forward(W, P, model)

            P_min = P.min(axis=0)
            P_max = P.max(axis=0) - P_min
            P_sum = (P - P_min).sum(axis=0)

            P_max = P_max / P_sum

            # P_max = P.max(axis=0)

            if converged is None:
                converged = P[:, P_max >= threshold]
            else:
                converged = np.hstack((converged, P[:, P_max >= threshold]))

            P = P[:, P_max < threshold]
            s = P.shape[1]
        if s != 0:
            has_converged = False
        P = converged
        print(f"t:{t}")
    else:
        for t in range(1, T + 1):
            W = np.array([gen.generate() for _ in range(samples)]).squeeze().T
            P = forward(W, P, model)

    choice_indices = P.argmax(axis=0)
    dist = np.array(np.bincount(choice_indices), dtype=np.double) / samples
    opts = model.M.shape[0]
    if len(dist) < opts:
        dist = np.append(dist, np.zeros(opts - len(dist)))

    if has_converged and sum(dist) != 1:
        print("error")

    return dist.reshape(-1, 1), has_converged


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
