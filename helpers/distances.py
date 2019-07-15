import numpy as np
import math


def exponential(o1, o2):
    return np.linalg.norm(o1 - o2)


def hotaling_matrix(b, dtype=np.double):
    H = np.array([[(b + 1) / 2, (b - 1) / 2], [(b - 1) / 2, (b + 1) / 2]], dtype=dtype)
    return H


def hotaling(o1, o2, b):
    H = hotaling_matrix(b)
    return (o2 - o1) @ H @ (o2 - o1).T


def hotaling_S_from_D(D, φ1, φ2):
    return np.eye(D.shape[0]) - φ2 * np.exp(-φ1 * D ** 2)


def hotaling_D(M, b):
    D = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            D[i][j] = hotaling(M[i], M[j], b)
    return D


def hotaling_S(M, φ1, φ2, b):
    D = hotaling_D(M, b)
    return hotaling_S_from_D(D, φ1, φ2)
