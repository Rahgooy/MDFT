"""
Learn MDFT using a recurrent neural network.
A dataset of different MDFT models is used to estimate common parameters used in those models.

Usage:
    learn_separate_m.py [options]

Examples:
    learn_separate_m.py --niter=100

Options:
    -h --help                  Show this screen.
    --niter=INT                Number of iterations. [default: 1000]
    --nprint=INT               Number of iterations per print. [default: 100]
    --ntest=INT                Number of test samples for evaluations[default: 10000]
    --ntrain=INT               Number of train samples. [default: 100]
    --i=STR                    input data set. [default: data/set4.json]
    --o=STR                    output path. [default: results/separate_m_learn_m.json]
    --m                        Learn M. [default: True]
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
from helpers.profiling import global_profiler
from separate_m_trainer import train


def get_options():
    opts = docpie(__doc__)
    # remove '--' in front of option names
    opts = {key.replace('--', ''): item for key, item in opts.items()}
    opts['niter'] = int(opts['niter'])
    opts['ntest'] = int(opts['ntest'])
    opts['ntrain'] = int(opts['ntrain'])
    opts['nprint'] = int(opts['nprint'])
    opts['m'] = eval(str(opts['m']))
    opts['w'] = eval(str(opts['w'])) and not opts['m']
    opts['s'] = eval(str(opts['s'])) and not opts['m']
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
        f, T = simMDF(D, m.C, M, w, data["threshold"], σ2, 10000)
        freq1[i] = f
        f, converged = get_threshold_based_dft_dist(m, 10000, data["threshold"], data["relative"])
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
    np.set_printoptions(precision=4, suppress=True)

    with open(opts['i'], 'r', encoding="UTF-8") as f:
        data = json.load(f)
    results = []
    for datasets in data['datasets']:
        print("*" * 70)
        if opts['m']:
            M_list = []
            best = {}
            for j in range(len(datasets['M'])):
                dataset = datasets.copy()
                dataset['M'] = [datasets['M'][j]]
                dataset['freq'] = [datasets['freq'][j]]
                best, it = train(dataset, opts)
                M_list.append(best['M'])

            best['M'] = M_list
            print("pred M:")
            M = np.vstack(M_list)
            print("[")
            for j in range(M.shape[0]):
                for i in M[j]:
                    print(f"{i:0.4f} ", end="")
                if j < M.shape[0] - 1:
                    print(";")
            print("]")
        else:
            best, it = train(datasets, opts)
            best['M'] = datasets['M']

        freq_list = []
        for m in best['M']:
            M = np.array(m)
            S = hotaling_S(M, best['φ1'], best['φ2'], best['b'])
            P0 = np.zeros((M.shape[0], 1))
            m = DFT(M, S, np.array(best['w']), P0, np.array(best['σ2']))
            f, converged = get_threshold_based_dft_dist(m, opts['ntest'], best["threshold"], datasets["relative"])
            freq_list.append(f.squeeze().tolist())

        sse = np.array(datasets['freq']) - np.array(freq_list)
        sse = (sse * sse).sum()

        best['freq'] = freq_list
        best['sse'] = sse
        best['actual-freq'] = datasets['freq']

        print("pred freq:")
        print(np.array(freq_list))
        print("Actual freq:")
        print(np.array(datasets['freq']))
        print(f"SSE: {sse:0.4f}")

        results.append(best)

    print(f"average SSE: {sum([s['sse'] for s in results]) / len(results)}")

    print(f"Time elapsed {time() - main_start:0.2f} seconds")

    global_profiler.print_profile()

    outPath = Path(opts['o'])
    outPath.parent.mkdir(exist_ok=True, parents=True)
    with outPath.open(mode='w') as f:
        results = {
            'nsamples': opts['ntrain'],
            'dataset': Path(opts['i']).name,
            'results': results,
            'ntest': opts['ntest'],
        }
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
