import numpy as np
from dft import get_fixed_T_dft_dist
from scipy import stats
from itertools import combinations, permutations


def dft_kl(model1, model2, samples, T):
    dist1 = get_fixed_T_dft_dist(model1, samples, T)
    dist2 = get_fixed_T_dft_dist(model2, samples, T)

    return stats.entropy(dist1, dist2), dist1.T, dist2.T


def dft_jsd(model1, model2, samples, T):
    dist1 = get_fixed_T_dft_dist(model1, samples, T)
    dist2 = get_fixed_T_dft_dist(model2, samples, T)

    return jsd(dist1, dist2), dist1.T, dist2.T


def jsd(dist1, dist2):
    """Jensen-Shanon Divergence"""
    M = (dist1 + dist2) / 2
    jsd = (stats.entropy(dist1, M) + stats.entropy(dist2, M)) / 2

    return jsd


def kendalltau_dist(a, b):
    tau = 0
    n_candidates = len(a)
    for i, j in combinations(range(n_candidates), 2):
        if (a[i] > a[j] and b[i] < b[j]) or (a[i] < a[j] and b[i] > b[j]):
            tau += 1
    total = n_candidates * (n_candidates - 1) / 2
    return tau / total
