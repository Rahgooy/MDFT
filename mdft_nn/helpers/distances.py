import numpy as np


def exponential(o1, o2):
    return np.linalg.norm(o1 - o2)


def gpd_H(attr_count, b):
    """returns the H matrix of Generalize Psychological Distance
    """
    n = attr_count

    def invB():
        ib = -np.ones((n, n))
        for i in range(1, n):
            ib[i-1, i] = n - 1
        ib[-1, :] = 1
        return ib / np.sqrt(n)
    B = invB()
    A = np.eye(n)
    A[-1, -1] = b
    H = B @ A @ B.T
    return H


def gpd(o1, o2, b):
    """returns the generalize psychological distance

    Args:
        o1 (vector): option 1
        o2 (vector): option 2
        b (number): b parameter

    Returns:
        number: generalize psychological distance
    """
    H = gpd_H(len(o1), b)
    return (o2 - o1) @ H @ (o2 - o1).T


def hotaling_S_from_D(D, phi1, phi2):
    return np.eye(D.shape[0]) - phi2 * np.exp(-phi1 * D ** 2)


def hotaling_D(M, b):
    D = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            D[i][j] = gpd(M[i], M[j], b)
    return D


def hotaling_S(M, phi1, phi2, b):
    D = hotaling_D(M, b)
    return hotaling_S_from_D(D, phi1, phi2)
