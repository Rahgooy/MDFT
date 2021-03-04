"""
Learn MDFT using a recurrent neural network.
A dataset of different MDFT models is used to estimate common parameters used in those models.

Usage:
    learn.py [options]

Examples:
    learn.py --niter=100

Options:
    -h --help                  Show this screen.
    --niter=INT                Number of iterations. [default: 150]
    --nprint=INT               Number of iterations per print. [default: 1]
    --ntest=INT                Number of test samples for evaluations[default: 10000]
    --ntrain=INT               Number of train samples. [default: 100]
    --i=STR                    input data set. [default: data/set-non-dominated-norm.mat]
    --o=STR                    output path. [default: results/Mw/set-non-dominated-norm]
    --m                        Learn M. [default: True]
    --w                        Learn W. [default: True]
    --s                        Learn S. [default: False]
"""
import json
from time import time

from docpie import docpie
import numpy as np
from pprint import pprint
from pathlib import Path
from helpers.profiling import global_profiler
from trainer import train
import mat4py

from trainer_helpers import get_model_dist


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


def load_data(opts):
    data = mat4py.loadmat(opts['i'])
    data = data['dataset']
    for d in data:
        d['idx'] = (np.array(d['idx']) - 1).tolist()  # adjust indexes to start from 0
        d['relative'] = False
    return data


def main():
    """Command line argument processing"""
    main_start = time()
    opts = get_options()
    pprint(opts)
    np.set_printoptions(precision=4, suppress=True)

    datasets = load_data(opts)
    results = []
    for d in datasets:
        print("*" * 90)
        start = time()
        best, it = train(d, opts)
        best['time'] = time() - start

        freq_list = get_model_dist(best, d, opts['ntest'])
        mse = np.array(d['D']) - np.array(freq_list)
        mse = (mse * mse).sum() / len(freq_list)

        best['freq'] = freq_list
        best['mse'] = mse
        best['actual_freq'] = d['D']

        print("pred freq:")
        print(np.array(freq_list))
        print("Actual freq:")
        print(np.array(d['D']))
        print(f"MSE: {mse:0.4f}")

        results.append(best)

    print(f"average MSE: {sum([s['mse'] for s in results]) / len(results)}")

    print(f"Time elapsed {time() - main_start:0.2f} seconds")

    global_profiler.print_profile()

    outPath = Path(opts['o'] + ".json")
    outPath.parent.mkdir(exist_ok=True, parents=True)
    with outPath.open(mode='w') as f:
        results = {
            'nsamples': opts['ntrain'],
            'dataset': Path(opts['i']).name,
            'results': results,
            'ntest': opts['ntest'],
        }
        json.dump(results, f, sort_keys=True, indent=4)
    outPath = Path(opts['o'] + ".mat")
    mat4py.savemat(str(outPath), results)


if __name__ == "__main__":
    main()
