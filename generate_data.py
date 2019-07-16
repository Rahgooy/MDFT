import json

from dft import get_fixed_T_dft_dist, get_threshold_based_dft_dist, DFT
import numpy as np
from helpers.distances import hotaling_S
from pathlib import Path


def generate_data(params):
    """
    Generate MDFT samples using specified parameters.
    """
    M, φ1, φ2, b, w, n_samples = params['M'], params['φ1'], params['φ2'], params['b'], params['w'], params['n_samples']
    T, tb, threshold, relative = params['T'], params['tb'], params['threshold'], params['relative']

    options = M.shape[0]
    P0 = np.zeros((options, 1))
    S = hotaling_S(M, φ1, φ2, b)
    model = DFT(M, S, w, P0)
    converged = True
    max_t = T
    if tb:
        dist, converged, max_t = get_threshold_based_dft_dist(model, n_samples, threshold, relative)
    else:
        dist = get_fixed_T_dft_dist(model, n_samples, T)

    return {
               "φ1": φ1,
               "φ2": φ2,
               "b": b,
               "dist": "hotaling",
               "tb": tb,
               "threshold": threshold,
               "relative": relative,
               "T": T,
               'freq': dist.squeeze().tolist(),
               'S': model.S.tolist(),
               'M': model.M.tolist(),
               'w': model.w.tolist(),
               'C': model.C.tolist(),
               'P0': model.P0.tolist(),
               'n_samples': n_samples
           }, converged, max_t


def generate_random_dataset(n_samples, n_datasets, n_options, path, tb, params):
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
    datasets = []
    for i in range(n_datasets):
        while True:
            p['threshold'] = params['threshold'] if 'threshold' in params else 0
            p['relative'] = params['relative'] if 'relative' in params else True
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

            data, converged, max_t = generate_data(p)
            if not converged:
                print("Not converged")
                continue
            if max_t < 10:
                print("Short deliberation")
                continue
            freq = np.array(data['freq'])
            print(freq)
            if sum(freq < min_freq_accepted) == sum(freq == 0):
                if (freq > 0).sum() >= min_non_zero_freq:
                    break
        datasets.append(data)
        print("=" * 50)
        print("data set number {} is generated. N_options : {}".format(i + 1, n_options))
        print("=" * 50)
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open(mode="w") as f:
        json.dump({
            'tb': tb,
            'datasets': datasets
        }, f, indent=4)


if __name__ == "__main__":
    params = {
        "φ1": 0.22,
        "φ2": 0.05,
        "b": 12,
        "w": 0.7,
        "threshold": 10,
        "relative": False,
        "T": 100,
    }
    generate_random_dataset(n_samples=5000,
                            n_datasets=10,
                            n_options=3,
                            path="data/threshold_o_3_absolute.json",
                            tb=True,
                            params=params)
