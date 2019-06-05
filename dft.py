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


def get_fixed_T_dft_dist(model, samples, T):
    dist, _, _ = __get_dft_dist(model, samples, False, T, 0)
    return dist


def get_threshold_based_dft_dist(model, samples, threshold):
    return __get_dft_dist(model, samples, True, 0, threshold)


def __get_dft_dist(model, samples, tb, T, threshold):
    def forward(w, prev_p, model):
        CM = model.C @ model.M
        V = CM @ w
        SP = model.S @ prev_p
        return SP + V

    gen = RouletteWheelGenerator(model.w)
    P = np.repeat(model.P0, samples, axis=1)
    has_converged = True
    MAX_T = 200
    MAX_TRIAL = 3
    max_t = 0
    min_t = MAX_T + 1
    if tb:
        s = samples
        converged = None

        t = 1
        trials = 0
        average_t = 0
        while s > 0 and trials < MAX_TRIAL:
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
            converged_count = sum(P_max >= threshold)
            average_t += t * converged_count
            if converged_count > 0:
                max_t = max(max_t, t)
                min_t = min(min_t, t)
            s = P.shape[1]
            t += 1
            if t >= MAX_T:  # reset long deliberations to generate new ones
                if s == samples:  # No sample converged
                    print(f"No sample converged after {MAX_T}.")
                    break
                P = np.repeat(model.P0, s, axis=1)
                print(f"Not converged after {MAX_T} : {s}. ")
                print(f"Reset trial number {trials}...")
                t = 0
                trials += 1
        if s != 0:
            has_converged = False
        P = converged
        average_t /= samples
        print(f"T - min: {min_t}, max: {max_t}, avg:{average_t}")
    else:
        for t in range(1, T + 1):
            W = np.array([gen.generate() for _ in range(samples)]).squeeze().T
            P = forward(W, P, model)

    choice_indices = P.argmax(axis=0)
    dist = np.array(np.bincount(choice_indices), dtype=np.double) / samples
    opts = model.M.shape[0]
    if len(dist) < opts:
        dist = np.append(dist, np.zeros(opts - len(dist)))

    return dist.reshape(-1, 1), has_converged, max_t