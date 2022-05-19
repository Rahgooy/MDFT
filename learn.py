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
    --nprint=INT               Number of iterations per print. [default: 30]
    --ntest=INT                Number of test samples for evaluations[default: 10000]
    --ntrain=INT               Number of train samples. [default: 100]
    --i=STR                    input data set. [default: data/sushi.json]
    --o=STR                    output path. [default: results/NN/pref_based/M/sushi]
    --m=STR                    Learn M. [default: True]
    --w=STR                    Learn W. [default: False]
    --s=STR                    Learn S. [default: False]
"""
import json
from time import time
from docpie import docpie
import numpy as np
from pprint import pprint
from pathlib import Path
import torch
import mat4py
import json

from mdft_nn.helpers.profiling import global_profiler
from mdft_nn.trainer import train
from mdft_nn.trainer_helpers import get_model_dist


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
    if opts['i'].endswith('json'):
        with open(opts['i']) as f:
            data = json.load(f)
    else:
        data = mat4py.loadmat(opts['i'])
        data = data['dataset']
        for d in data:
            # adjust indexes to start from 0
            d['idx'] = (np.array(d['idx']) - 1)
            d['pref_based'] = d['pref_based'] == 1
            if d['idx'].ndim == 1:
                d['idx'] = [d['idx'].tolist()]
                d['D'] = [d['D']]
            else:
                d['idx'] = d['idx'].tolist()
    return data


def main():
    """Command line argument processing"""
    np.random.seed(100)
    torch.manual_seed(100)
    main_start = time()
    opts = get_options()
    pprint(opts)
    np.set_printoptions(precision=4, suppress=True)

    datasets = load_data(opts)
    results = []
    for i, d in enumerate(datasets):
        print("*" * 90)
        print(f"* #{i+1}")
        start = time()
        best, it = train(d, opts)
        best['time'] = time() - start

        freq_list = get_model_dist(best, d, opts['ntest'])
        mse = 0
        nc = len(d['D'])
        for i in range(len(d['D'])):
            m = np.array(d['D'][i]) - np.array(freq_list[i])
            mse += (m*m).sum() / nc

        best['freq'] = freq_list
        best['mse'] = mse
        best['actual_freq'] = d['D']

        print("pred freq:")
        for row in freq_list:
            print(np.array(row))
        print("Actual freq:")
        for row in d['D']:
            print(np.array(row))
        print(f"MSE: {mse:0.4f}")

        results.append(best)

    print(f"average MSE: {sum([s['mse'] for s in results]) / len(results)}")

    print(f"Time elapsed {time() - main_start:0.2f} seconds")

    out_path = Path(opts['o'] + ".json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open(mode='w') as f:
        results = {
            'nsamples': opts['ntrain'],
            'dataset': Path(opts['i']).name,
            'results': results,
            'ntest': opts['ntest'],
        }
        json.dump(results, f, sort_keys=True, indent=4)
    out_path = Path(opts['o'] + ".mat")
    mat4py.savemat(str(out_path), results)


if __name__ == "__main__":
    main()
