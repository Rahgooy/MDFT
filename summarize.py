import json
from pathlib import Path
import numpy as np
import mat4py

from helpers.evaluation import kendalltau_dist, get_attr_index, jsd

baseDir = Path('results/MLE')


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
        w = np.array(d['w'])
        w_ = np.array(r['w'])
        idx = get_attr_index(M, M_, w, w_)
        re_order = idx[0]

        r['kt1'] = kendalltau_dist(M[:, 0], M_[:, idx[0]])
        r['kt2'] = kendalltau_dist(M[:, 1], M_[:, idx[1]])

        w = np.array(d['w']).squeeze()
        w_ = np.array(r['w']).squeeze()
        w = w[idx]
        w_ = w_[idx]

        r['jsd'] = np.mean([jsd(d['D'][j], r['freq'][j]) for j in range(len(d['D']))])
        r['w_jsd'] = jsd(w, w_)
        r['re_order'] = re_order


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
line_len = 124
for p in summary:
    print(" " + "=" * line_len)
    print(f"|{'learn ' + p:^123s} |")
    print(" " + "-" * line_len)
    print(f"| {'Set':30s}|{'MSE':^14s}  |  {'JSD Choice':^14s}  |  {'JSD W':^14s}  |  {'kt mean':^14s}  |  "
          f"{'time':^14s}  |")
    print(f"| {'':30s}|{'mean':^7s} {'std':^7s} |  {'mean':^7s} {'std':^7} |  {'mean':^7s} {'std':^7} |  "
          f"{'mean':^7s} {'std':^7} |  {'mean':^7s} {'std':^7} |")
    print(" " + "-" * line_len)
    for s in sorted(summary[p].keys()):
        kt = (summary[p][s]['kt1'] + summary[p][s]['kt2']) / 2
        print(f"| {s:30s}|{summary[p][s]['mse'].mean():<0.5f} {summary[p][s]['mse'].std():>0.5f} |  "
              f"{summary[p][s]['jsd'].mean():<0.5f} {summary[p][s]['jsd'].std():<0.5f} |  "
              f"{summary[p][s]['w_jsd'].mean():<0.5f} {summary[p][s]['w_jsd'].std():<0.5f} |  "
              f"{kt.mean():<0.5f} {kt.std():<0.5f} |  "
              f"{summary[p][s]['time'].mean():^7.2f} {summary[p][s]['time'].std():^7.2f} |"
              )
    print(" " + "-" * line_len)
    print("\n")
