"""
Learn MDFT using a recurrent neural network.
A dataset of different MDFT models is used to estimate common parameters used in those models.

Usage:
    train_batch.py [options]

Examples:
    train_batch.py --niter=100

Options:
    -h --help                  Show this screen.
    --niter=INT                Number of iterations. [default: 100]
    --nprint=INT               Number of iterations per print. [default: 10]
    --ntest=INT                Number of test samples for evaluations[default: 500]
    --ntrain=INT               Number of train samples. [default: 20]
    --i=STR                    input data set. [default: data/set4.json]
    --o=STR                    output path. [default: results/learn_m_pref_origw_single.txt]
    --m                        Learn M. [default: False]
    --w                        Learn W. [default: True]
    --s                        Learn S. [default: False]
"""
import json
from time import time

from busemeyer.distfunct import distfunct
from busemeyer.simMDF import simMDF
from dft import DFT, get_threshold_based_dft_dist
from helpers.distances import hotaling_S
from docpie import docpie
import numpy as np
from pprint import pprint
from pathlib import Path
from helpers.evaluation import dft_kl
from trainer import train


def get_options():
    opts = docpie(__doc__)
    # remove '--' in front of option names
    opts = {key.replace('--', ''): item for key, item in opts.items()}
    opts['niter'] = int(opts['niter'])
    opts['ntest'] = int(opts['ntest'])
    opts['ntrain'] = int(opts['ntrain'])
    opts['nprint'] = int(opts['nprint'])
    opts['m'] = eval(str(opts['m']))
    opts['w'] = eval(str(opts['w']))
    opts['s'] = eval(str(opts['s']))
    return opts


def simulate(data, opts):
    MM = np.array(data['M'])
    φ1 = data['φ1']
    φ2 = data['φ2']
    b = data['b']
    σ2 = data['sigma2']
    freq1 = np.zeros((MM.shape[0], MM.shape[1]))
    freq2 = np.zeros((MM.shape[0], MM.shape[1]))
    for i in range(MM.shape[0]):
        M = MM[i, :, :]
        S = hotaling_S(M, φ1, φ2, b)
        D, _ = distfunct(M, b, φ1, φ2)
        w = np.array(data['w'])
        P0 = np.zeros((M.shape[0], 1))
        m = DFT(M, S, w, P0, σ2)
        f, T = simMDF(D, m.C, M, w, data["threshold"], σ2, 30000)
        freq1[i] = f
        f, converged = get_threshold_based_dft_dist(m, 30000, data["threshold"], data["relative"])
        freq2[i] = f.T
    print("M:")
    print(MM)
    print("w: ")
    print(w)
    print("S params:")
    print(φ1, φ2, b)
    print("S:")
    print(S)
    print("Simulations")
    print(freq1)
    print(freq2)
    print(freq1 - freq2)
    print(np.abs(freq1 - freq2).max())
    print("Actual")
    print(np.array(data['freq']) - freq1)
    print(np.abs(np.array(data['freq']) - freq1).max())
    print(np.abs(np.array(data['freq']) - freq2).max())


def main():
    """Command line argument processing"""
    main_start = time()
    opts = get_options()
    pprint(opts)

    with open(opts['i'], 'r', encoding="UTF-8") as f:
        data = json.load(f)
    datasets = data['datasets'][5]

    # simulate(datasets[0], opts)
    # if not check_data(datasets, opts):
    #     return
    best, it = train(datasets, opts)
    print(f"Time elapsed {time() - main_start:0.2f} seconds")
    return
    outPath = Path(output)
    outPath.parent.mkdir(exist_ok=True, parents=True)
    with outPath.open(mode='w') as f:
        data.summary(f)
        print("============================= Settings ==============================", file=f)
        print("Number of Iterations : {}[max{}, best{}]".format(it + 1, niter, best["iter"]), file=f)
        print("Learning rate : {}".format(lr), file=f)
        print("Optimizer: {}".format(optim_name), file=f)
        print("============================= Results  ==============================", file=f)
        time_ = time() - main_start
        print("Time : {:0.3f}s".format(time_))
        print("Time : {:0.3f}s".format(time_), file=f)
        print("{}: {}".format(loss_name, best["error"]), file=f)
        print("M:", file=f)
        print(np.array(best["M"]), file=f)
        print("φ1: {}".format(best["φ1"]), file=f)
        print("φ2: {}".format(best["φ2"]), file=f)
        print("w: {}".format(best["w"]), file=f)

        model_M = np.array(best["M"])
        model_S = hotaling_S(model_M, best["φ1"], best["φ2"], model.b)
        model_dft = DFT(model_M, model_S, np.array(best["w"]), model.P0)
        kl, actual_dist, model_dist = dft_kl(data, model_dft, ntest, T)
        avg = np.array([d.choice for d in data.samples])
        avg = np.average(avg, axis=0)

        print("Test sample size: {}".format(ntest), file=f)
        print("Actual dist: {}".format(actual_dist), file=f)
        print("Model dist: {}".format(model_dist), file=f)
        actual_freq = ["{:04d}".format(int(x * ntest)) for x in actual_dist.squeeze()]
        print("Actual freq: [{}]".format(' & '.join(actual_freq)),
              file=f)
        model_freq = ["{:04d}".format(int(x * ntest)) for x in model_dist.squeeze()]
        print("Model freq: [{}]".format(' & '.join(model_freq)),
              file=f)
        print("KL-Divergence: {}".format(kl), file=f)

        print("Test sample size: {}".format(ntest))
        print("Actual dist: {}".format(actual_dist))
        print("Model dist: {}".format(model_dist))
        print("KL-Divergence: {}".format(kl))

    outPath = outPath.parent / (outPath.name[:-3] + "json")
    with outPath.open(mode='w') as f:
        avg = np.array([d.choice for d in data.samples])
        avg = np.average(avg, axis=0).tolist()

        results = {
            'nsamples': len(data.samples),
            'data': {
                'M': data.M.tolist(),
                'params': data.parameters,
                'S': data.S.tolist(),
                'w': data.w.tolist(),
                'dist': avg,
            },
            'iter': it + 1,
            'max_iter': niter,
            'best': best,
            'lr': lr,
            'optim': optim_name,
            'time': time_,
            'loss_name': loss_name,
            'ntest': ntest,
            'actual_dist': actual_dist.tolist(),
            'model_dist': model_dist.tolist(),
            'actual_freq': actual_freq,
            'model_freq': model_freq
        }
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
