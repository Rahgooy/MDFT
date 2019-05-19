from dft import generate_fixed_time_DFT_samples, save_DFT_dataset, generate_threshold_based_DFT_samples
import numpy as np
from helpers.distances import hotaling_S_from_D, hotaling, exp_S_from_D, exponential
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
    T, min_freq_accepted, min_non_zero_freq = params['T'], params['min_freq_accepted'], params['min_non_zero_freq']
    tb, threshold, p, post, save = params['tb'], params['threshold'], params['path'], params['post'], params['save']

    options = M.shape[0]
    D = np.zeros((options, options))
    P0 = np.zeros((options, 1))
    dist_type = "hotaling"
    parameters = {
        "φ1": φ1,
        "φ2": φ2,
        "dist": dist_type,
        "tb": tb,
        "threshold": threshold,
        "T": T
    }
    if dist_type == "exp":
        for i in range(options):
            for j in range(options):
                D[i][j] = exponential(M[i], M[j])
        S = exp_S_from_D(D, φ1, φ2)
    else:
        for i in range(options):
            for j in range(options):
                D[i][j] = hotaling(M[i], M[j], b)
        S = hotaling_S_from_D(D, φ1, φ2)
        parameters["b"] = b
    if tb:
        data = generate_threshold_based_DFT_samples(M, S, w, P0, n_samples, threshold, parameters)
    else:
        data = generate_fixed_time_DFT_samples(M, S, w, P0, n_samples, T, parameters)

    avg = np.array([d.choice for d in data.samples])
    avg = np.average(avg, axis=0)
    np.set_printoptions(precision=4, suppress=True)

    print(avg.T.squeeze())
    print("")
    if (avg > min_freq_accepted).sum() < min_non_zero_freq or (avg < min_freq_accepted).sum() != (avg == 0).sum():
        return False

    if save:
        Path(p).mkdir(exist_ok=True, parents=True)
        save_DFT_dataset(data, "{}set_{}_n{}_l{}_o{}{}.pickle".format(p, dist_type, n_samples, T, options, post))

    return True, data


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
    for i in range(n_datasets):
        min_non_zero_freq = 2
        min_freq_accepted = 0.1
        while True:
            length = params['T'] if 'T' in params else np.random.randint(50, 100)
            M = params['M'] if 'M' in params else np.random.uniform(1.0, 5.0, (n_options, 2))
            b = params['b'] if 'b' in params else 10
            φ1 = params['φ1'] if 'φ1' in params else 0.02
            φ2 = params['φ2'] if 'φ2' in params else 0.02
            if 'w' in params:
                w = params['w']
            else:
                w = [0.3, 0.5, 0.7][np.random.randint(0, 3)]
                w = np.array([[w], [1 - w]])

            success = generate_data(M, n_samples, length, φ1, φ2, w, b, min_non_zero_freq, min_freq_accepted, path,
                                    "_{}".format(i + 1))
            if success:
                break
        print("data set number {} is generated. N_options : {}".format(i + 1, n_options))


if __name__ == "__main__":
    for s in [20, 30, 50, 100, 150]:
        generate_random_data(n_samples=s, n_datasets=100, n_options=9, path="data/random/set_o_9/")
