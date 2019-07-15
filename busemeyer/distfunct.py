import numpy as np


def distfunct(M, b, phi1, phi2):
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

    D = - phi2 * np.exp(-phi1 * D ** 2)  # gamma

    EG = np.linalg.eig(D)
    return D, EG
