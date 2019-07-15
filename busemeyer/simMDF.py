from math import sqrt
import numpy as np


def simMDF(G3, C3, M3, w, theta1, sig2, Ns):
    nd = M3.shape[0]  # number of options
    V3 = C3 @ M3 @ w  # mean valence
    P3 = np.zeros(nd)  # initial value of counter
    T = 0  # initial value for time

    P = np.zeros((nd, Ns))
    B = np.zeros(Ns)
    n = Ns
    while (B < theta1).any():
        W = np.random.rand(1, n) < w[0]
        W = np.vstack((W, 1 - W))
        E3 = C3 @ M3 @ (W - w) + sig2 * C3 @ np.random.randn(nd, n)  # compute noise
        P = P + G3 @ P + V3 + E3  # accumulate
        B = P.max(axis=0)
        converged = B >= theta1
        if converged.any():
            Ind = P[:, converged].argmax(axis=0)
            P3 = P3 + np.array([(Ind == 0).sum(), (Ind == 1).sum(), (Ind == 2).sum()])
        T = T + n
        P = P[:, B < theta1]
        n = P.shape[1]

    P3 = P3 / Ns
    T = T / Ns

    return P3, T


def simMDF_orig(G3, C3, M3, w, theta1, sig2, Ns):
    # function[P3 T] = simMDF(G3, C3, M3, w, theta1, sig2, Ns)
    # S3 is the n x n gamma feedback matrix
    # C3 is a contrast matrix
    # M3 is a n x m matrix of values of each option(row) on each attribute(col)

    h = 1
    hh = sqrt(h)  # time unit(canbe changed)
    na = w.shape[0]
    Na = np.arange(na).reshape((na, 1))
    # don't need to change
    nd = M3.shape[0]  # number of options
    V3 = C3 @ M3 @ w  # mean valence
    P3 = np.zeros(nd)  # initial value of counter
    T = 0  # initial value for time

    for ns in range(Ns):
        B = 0
        t = 0
        P = np.zeros((nd, 1))
        Ind = 0
        while B < theta1:
            W = int(w[0] > np.random.rand())
            W = np.array([[W], [1 - W]])  # pick an attribute
            # W = (WV == Na);
            E3 = C3 @ M3 @ (W - w * h) + sig2 * C3 @ np.random.randn(nd, 1)  # compute noise
            P = P + G3 @ P * h + V3 * h + E3 * hh  # accumulate
            Ind = P.argmax()  # find max
            B = P[Ind]
            t = t + h  # track time
            # end of one simulation
        P3 = P3 + np.array([Ind == 0, Ind == 1, Ind == 2])
        T = T + t
        # end of all N simulations

    P3 = P3 / Ns
    T = T / Ns

    return P3, T
