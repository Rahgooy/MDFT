import json

from dft import get_fixed_T_dft_dist, get_threshold_based_dft_dist, DFT
import numpy as np
from helpers.distances import hotaling_S
from pathlib import Path


def generate_data(params):
    """
    Generate MDFT samples using specified parameters.
    Parameters include:
        tb: threshold based MDFT
        threshold:
        M:
        n_samples:
        T:
        φ1:
        φ2:
        w:
        b:
        min_non_zero_freq:
        min_freq_accepted:
        p:
        post:
        save:
    :return:
    """
    M, φ1, φ2, b, w, n_samples = params['M'], params['φ1'], params['φ2'], params['b'], params['w'], params['n_samples']
    T, tb, threshold = params['T'], params['tb'], params['threshold']

    options = M.shape[0]
    P0 = np.zeros((options, 1))
    S = hotaling_S(M, φ1, φ2, b)
    model = DFT(M, S, w, P0)
    converged = True
    if tb:
        dist, converged = get_threshold_based_dft_dist(model, n_samples, threshold)
    else:
        dist = get_fixed_T_dft_dist(model, n_samples, T)

    return {
               "φ1": φ1,
               "φ2": φ2,
               "b": b,
               "dist": "hotaling",
               "tb": tb,
               "threshold": threshold,
               "T": T,
               'freq': dist.squeeze().tolist(),
               'S': model.S.tolist(),
               'w': model.w.tolist(),
               'C': model.C.tolist(),
               'P0': model.P0.tolist(),
               'n_samples': n_samples
           }, converged


def generate_random_data(n_samples, n_datasets, n_options, path, tb, params):
    """
    Generates MDFT datasets with random parameters.
    :param n_samples: number of samples in each dataset
    :param n_datasets: number of datasets to be generated
    :param n_options: number of options
    :param path: the output path
    :param tb: threshold based MDFT if true and fixed-time MDFT if false
    :param params: MDFT parameters. Each parameter is generated randomly unless a value is provided using this variable.
                   The parameters include:
                   'T': deliberation time
                   'φ1':
                   'φ2':
                   'b':
                   'w':
                   'M':
    :return:
    """
    min_non_zero_freq = 2
    min_freq_accepted = 0.1
    p = {
        'tb': tb,
        'n_samples': n_samples
    }
    dataset = []
    for i in range(n_datasets):
        while True:
            p['threshold'] = params['threshold'] if 'threshold' in params else 0
            p['T'] = params['T'] if 'T' in params else np.random.randint(50, 100)
            p['M'] = params['M'] if 'M' in params else np.random.uniform(1.0, 5.0, (n_options, 2))
            p['b'] = params['b'] if 'b' in params else 10
            p['φ1'] = params['φ1'] if 'φ1' in params else 0.02
            p['φ2'] = params['φ2'] if 'φ2' in params else 0.02
            if 'w' in params:
                w = params['w']
                p['w'] = np.array([[w], [1 - w]])
            else:
                w = [0.3, 0.5, 0.7][np.random.randint(0, 3)]
                p['w'] = np.array([[w], [1 - w]])

            data, converged = generate_data(p)
            if not converged:
                print("Not converged")
                continue
            freq = np.array(data['freq'])
            print(freq)
            if sum(freq < min_freq_accepted) == sum(freq == 0):
                if (freq > 0).sum() >= min_non_zero_freq:
                    break
        dataset.append(data)
        print("data set number {} is generated. N_options : {}".format(i + 1, n_options))
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open(mode="w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    params = {
        "φ1": 0.22,
        "φ2": 0.05,
        "b": 12,
        "w": 0.5,
        "threshold": 0.7,
        "T": 100,
    }
    generate_random_data(n_samples=5000,
                         n_datasets=10,
                         n_options=3,
                         path="data/threshold_o_3.json",
                         tb=True,
                         params=params)
