import numpy as np
from mdft_nn.mdft import get_time_based_dft_dist
from scipy import stats
from itertools import combinations


def dft_kl(model1, model2, samples, T):
    dist1 = get_time_based_dft_dist(model1, samples, T)
    dist2 = get_time_based_dft_dist(model2, samples, T)

    return stats.entropy(dist1, dist2), dist1.T, dist2.T


def dft_jsd(model1, model2, samples, T):
    dist1 = get_time_based_dft_dist(model1, samples, T)
    dist2 = get_time_based_dft_dist(model2, samples, T)

    return jsd(dist1, dist2), dist1.T, dist2.T


def jsd(dist1, dist2, eps=0.0):
    """Jensen-Shanon Divergence"""
    dist1 = np.array(dist1) + eps
    dist1 /= dist1.sum()
    dist2 = np.array(dist2) + eps
    dist2 /= dist2.sum()

    M = (dist1 + dist2) / 2
    jsd = (stats.entropy(dist1, M) + stats.entropy(dist2, M)) / 2

    return jsd


def kendalltau_dist(a, b):
    a = np.argsort(a).tolist()
    b = np.argsort(b).tolist()
    tau = 0
    n_candidates = len(a)
    for i, j in combinations(range(n_candidates), 2):
        if (a[i] > a[j] and b[i] < b[j]) or (a[i] < a[j] and b[i] > b[j]):
            tau += 1
    total = n_candidates * (n_candidates - 1) / 2
    return tau / total


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=np.float)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    x, y = stats.t.interval(confidence, n - 1, loc=m, scale=se)
    return m, se, m - x


def get_attr_index(m, m_):
    kt1 = min(kendalltau_dist(m[:, 0], m_[:, 0]),
              kendalltau_dist(m[:, 1], m_[:, 1]))
    kt2 = min(kendalltau_dist(m[:, 0], m_[:, 1]),
              kendalltau_dist(m[:, 1], m_[:, 0]))
    if kt1 <= kt2:
        return [0, 1]
    return [1, 0]
