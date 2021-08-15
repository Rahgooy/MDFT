import json
from os import name
from pathlib import Path
import numpy as np
import mat4py
import scipy

from helpers.evaluation import kendalltau_dist, get_attr_index, jsd


def load_data(path):
    data = mat4py.loadmat(path)
    return data


def set_evaluations(results):
    dataset = load_data("data/" + results['dataset'])
    dataset = dataset['dataset']
    for i in range(len(dataset)):
        d = dataset[i]
        r = results['results'][i]
        M = np.array(d['M'])
        M_ = np.array(r['M'])
        idx = get_attr_index(M, M_)
        re_order = idx[0]

        r['kt1'] = kendalltau_dist(M[:, 0], M_[:, idx[0]])
        r['kt2'] = kendalltau_dist(M[:, 1], M_[:, idx[1]])

        w = np.array(d['w']).squeeze()
        w_ = np.array(r['w']).squeeze()
        w = w[idx]
        w_ = w_[idx]

        dist1 = np.array(d['D']).reshape(-1, M.shape[0])
        dist2 = np.array(r['freq']).reshape(-1, M.shape[0])

        r['jsd'] = np.mean([jsd(dist1[j], dist2[j])
                            for j in range(len(dist1))])
        r['w_jsd'] = jsd(w, w_)
        r['re_order'] = re_order


def summarize(model):
    baseDir = Path(f'results/{model}/')
    summary = {}
    for dir in baseDir.iterdir():
        if dir.is_dir():
            param_name = dir.name
            summary[param_name] = {}
            for s in dir.iterdir():
                if s.is_file() and s.name.endswith(".mat"):
                    set_name = s.name[:-4]
                    # with s.open(mode='r') as f:
                    data = load_data(str(s))  # json.load(f)
                    set_evaluations(data)
                    summary[param_name][set_name] = {
                        'mse': np.array([d['mse'] for d in data['results']]),
                        'jsd': np.array([d['jsd'] for d in data['results']]),
                        'w_jsd': np.array([d['w_jsd'] for d in data['results']]),
                        'kt1': np.array([d['kt1'] for d in data['results']]),
                        'kt2': np.array([d['kt2'] for d in data['results']]),
                        'time': np.array([d['time'] for d in data['results']]),
                    }

    np.set_printoptions(precision=4, suppress=False, linewidth=200)

    line_len = 132
    for p in summary:
        print(" " + "=" * line_len)
        print(f"|{f'learn {p}[{model}]':^131s} |")
        print(" " + "-" * line_len)
        print(f"| {'Set':38s}|{'MSE':^14s}  |  {'JSD Choice':^14s}  |  {'JSD W':^14s}  |  {'kt mean':^14s}  |  "
              f"{'time':^14s}  |")
        print(f"| {'':38s}|{'mean':^7s} {'sem':^7s} |  {'mean':^7s} {'sem':^7} |  {'mean':^7s} {'sem':^7} |  "
              f"{'mean':^7s} {'sem':^7} |  {'mean':^7s} {'sem':^7} |")
        print(" " + "-" * line_len)
        for s in sorted(summary[p].keys()):
            kt = (summary[p][s]['kt1'] + summary[p][s]['kt2']) / 2
            print(f"| {s:38s}|{summary[p][s]['mse'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['mse']):>0.5f} |  "
                  f"{summary[p][s]['jsd'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['jsd']):<0.5f} |  "
                  f"{summary[p][s]['w_jsd'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['w_jsd']):<0.5f} |  "
                  f"{kt.mean():<0.5f} {scipy.stats.sem(kt):<0.5f} |  "
                  f"{summary[p][s]['time'].mean():^7.2f} {scipy.stats.sem(summary[p][s]['time']):^7.2f} |"
                  )
        print(" " + "-" * line_len)
        print("\n")


summarize('MLE')
summarize('NN')
