import numpy as np


class MDFT:
    def __init__(self, M, S, w, P0, sig2=1):
        self.M = M
        self.S = S
        self.w = w
        self.P0 = P0
        self.P = [P0]
        self.sig2 = sig2
        self.V = [np.zeros(P0.shape)]
        self.W = [np.zeros(w.shape)]
        self.t = 0
        self.C = np.ones((M.shape[0], M.shape[0])) * -(1 / (M.shape[0] - 1))
        for i in range(len(M)):
            self.C[i][i] = 1


def get_fixed_T_dft_dist(model, samples, T):
    dist, _ = __get_dft_dist(model, samples, False, T, 0)
    return dist


def get_threshold_based_dft_dist(model, samples, threshold, relative=True):
    return __get_dft_dist(model, samples, True, 0, threshold, relative)


def __get_dft_dist(model, samples, tb, T, threshold, relative=True):
    def forward(C, CM, W, S, P, sig2):
        V = CM @ W
        E = sig2 * C @ np.random.randn(P.shape[0], P.shape[1])
        SP = S @ P
        return SP + V + E

    def get_max_pref(P, relative):
        if relative:
            P_min = P.min(axis=0)
            P_max = P.max(axis=0) - P_min
            P_sum = (P - P_min).sum(axis=0)
            P_max = P_max / P_sum
        else:
            P_max = P.max(axis=0)
        return P_max

    P = np.repeat(model.P0, samples, axis=1)
    has_converged = True
    MAX_T = 100000
    CM = model.C @ model.M
    if tb:
        n = samples
        converged = None
        t = 1
        while n > 0 and t < MAX_T:
            W = np.random.binomial(1, model.w[0], n)
            W = np.vstack((W, 1.0 - W))
            P = forward(model.C, CM, W, model.S, P, model.sig2)
            P_max = get_max_pref(P, relative)

            if converged is None:
                converged = P[:, P_max >= threshold]
            else:
                converged = np.hstack((converged, P[:, P_max >= threshold]))

            P = P[:, P_max < threshold]
            n = P.shape[1]
            t += 1

        has_converged = n == 0
        P = converged
    else:
        for t in range(1, T + 1):
            W = np.random.binomial(1, model.w[0], samples)
            W = np.vstack((W, 1.0 - W))
            P = forward(model.C, CM, W, model.S, P, model.sig2)

    choice_indices = P.argmax(axis=0)
    dist = np.array(np.bincount(choice_indices, minlength=P.shape[0]), dtype=np.double) / samples

    return dist.reshape(-1, 1), has_converged
