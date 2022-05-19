import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import mat4py
from numpy.core.defchararray import index
import scipy
from matplotlib import pyplot as plt
from mdft_nn.trainer_helpers import get_model_dist
from mdft_nn.helpers.evaluation import kendalltau_dist, get_attr_index, jsd


def load_data(path):
    if path.endswith('json'):
        with open(path) as f:
            data = json.load(f)
    else:
        data = mat4py.loadmat(path)
    if 'results' in data and type(data['results']) != list:
        data['results'] = [data['results']]
    return data


def set_evaluations(results, type_):
    dataset = load_data(f"data/{type_}/" + results['dataset'])
    dataset = dataset['dataset'] if 'dataset' in dataset else dataset
    for i in range(len(dataset)):
        d = dataset[i]
        r = results['results'][i]
        r['data'] = d
        M = np.array(d['M'])
        M_ = np.array(r['M'])
        idx = get_attr_index(M, M_)
        re_order = idx[0]

        r['kt1'] = kendalltau_dist(M[:, 0], M_[:, idx[0]])
        r['kt2'] = kendalltau_dist(M[:, 1], M_[:, idx[1]])
        r['kt'] = (r['kt1'] + r['kt2']) / 2

        w = np.array(d['w']).squeeze()
        w_ = np.array(r['w']).squeeze()
        w = w[idx]
        w_ = w_[idx]

        dist1 = np.array(d['D']) if np.array(
            d['D']).ndim > 1 else np.array([d['D']])
        dist2 = np.array(r['freq']) if np.array(
            r['freq']).ndim > 1 else np.array([r['freq']])

        r['jsd'] = np.mean([jsd(dist1[j], dist2[j], eps=0.0)
                            for j in range(len(dist1))])
        r['w_jsd'] = jsd(w, w_, eps=1e-6)
        r['re_order'] = re_order

        r['no'] = len(d['idx']) if np.ndim(d['idx']) == 1 else len(d['idx'][0])
        r['ncombs'] = 1 if np.ndim(d['idx']) == 1 else len(d['idx'])
        r['nopts'] = M.shape[0]


def print_summary(summary, model, type):
    np.set_printoptions(precision=4, suppress=False, linewidth=200)
    line_len = 140
    for p in summary:
        print(" " + "=" * line_len)
        print(f"|{f'learn {p}[{model}-{type}]':^139s} |")
        print(" " + "-" * line_len)
        print(f"| {'Set':46s}|{'Subset-JSD':^14s}  |  {'JSD Choice':^14s}  |  {'JSD W':^14s}  |  {'kt mean':^14s}  |  "
              f"{'whole_jsd':^14s}  |")
        print(f"| {'':46s}|{'mean':^7s} {'sem':^7s} |  {'mean':^7s} {'sem':^7} |  {'mean':^7s} {'sem':^7} |  "
              f"{'mean':^7s} {'sem':^7} |  {'mean':^7s} {'sem':^7} |")
        print(" " + "-" * line_len)
        for s in sorted(summary[p].keys()):
            kt = (summary[p][s]['kt1'] + summary[p][s]['kt2']) / 2
            print(f"| {s:46s}|{summary[p][s]['subsets_jsd'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['subsets_jsd']):>0.5f} |  "
                  f"{summary[p][s]['jsd'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['jsd']):<0.5f} |  "
                  f"{summary[p][s]['w_jsd'].mean():<0.5f} {scipy.stats.sem(summary[p][s]['w_jsd']):<0.5f} |  "
                  f"{kt.mean():<0.5f} {scipy.stats.sem(kt):<0.5f} |  "
                  f"{summary[p][s]['whole_jsd'].mean():^0.5f} {scipy.stats.sem(summary[p][s]['whole_jsd']):^0.5f} |"
                  )
        print(" " + "-" * line_len)
        print("\n")


def extract_metric(m, metric, nsamples=None, x_val=lambda x: x['no'], no=None, ncomb=1):
    vals = [(x_val(m[d]), m[d][metric].mean(), scipy.stats.sem(m[d][metric]))
            for d in m if (m[d]['ncombs'] == ncomb) and
            (nsamples is None or m[d]['nsamples'] == nsamples) and
            (no is None or m[d]['no'] == no)]
    vals = sorted(vals, key=lambda x: x[0])
    x = [x[0] for x in vals]
    y = [x[1] for x in vals]
    err = [x[2] for x in vals]

    return x, y, err


def plot_metric(jsd_data, type, yscale, metric, ylabel, name='options', legend=True,
                legend_loc='upper left'):
    x_vals = [20, 30, 50, 100, 150] if name == 'samples' else [3, 5, 7, 10]
    w = 0.7
    for param in jsd_data:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for i, d in enumerate(jsd_data[param]):
            idx = np.array([x_vals.index(x) for x in d['x']]) * 2 + 1.0
            idx += i * w - w/2 if len(jsd_data[param]) > 1 else 0
            label = 'RNN' if d['model'] == 'NN' else d['model']
            plt.bar(idx, d['y'], yerr=d['err'],
                    label=label, width=w, error_kw=dict(
                        lw=2, capsize=3, ecolor='#111'),
                    color=['#007acc', '#7acc00'][i], edgecolor='#111', linewidth=1.5,
                    hatch=['\\\\', '//'][i])

        # Two unit space between bars. One unit padding and start from 0
        idx = np.arange(1, len(x_vals)*2 + 1, 2).tolist()
        plt.xticks([0] + idx + [idx[-1] + 1], labels=[''] + x_vals + [''])
        plt.yscale(yscale)
        plt.ylabel(ylabel, fontweight='black', fontfamily='Arial')
        xlabel = 'Number of options' if name == 'options' else 'Training sample size'
        plt.xlabel(xlabel, fontweight='black', fontfamily='Arial')
        ax_style(ax)

        if legend:
            plt.legend(loc=legend_loc)
        plt.tight_layout()
        plt.savefig(f'./results/figures/{name}-{param}-{type}-{metric}.pdf')
        plt.close()


def ax_style(ax):
    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlim(0,8)

    yticks = ax.get_yticks().tolist()
    ax.set_ylim(yticks[0], ax.get_ylim()[1])

    xticks = ax.get_xticks().tolist()
    ax.set_xlim(xticks[0], xticks[-1])

    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))


def style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'


def option_size_plot(summary_list, type):
    w_jsd_data = defaultdict(list)
    jsd_data = defaultdict(list)
    jsd_subsets_data = defaultdict(list)
    kt_data = defaultdict(list)
    time_data = defaultdict(list)
    for model in summary_list:
        model_summary = summary_list[model]
        for param in model_summary:
            x, y, err = extract_metric(
                model_summary[param], 'jsd', 100 if model == 'NN' else 5000)
            jsd_data[param].append(
                {'x': x, 'y': y, 'err': err, 'model': model})

            x, y, err = extract_metric(
                model_summary[param], 'subsets_jsd', 100 if model == 'NN' else 5000)
            jsd_subsets_data[param].append(
                {'x': x, 'y': y, 'err': err, 'model': model})

            x, y, err = extract_metric(
                model_summary[param], 'w_jsd', 100 if model == 'NN' else 5000)
            w_jsd_data[param].append(
                {'x': x, 'y': y, 'err': err, 'model': model})

            x, y, err = extract_metric(
                model_summary[param], 'kt', 100 if model == 'NN' else 5000)
            kt_data[param].append(
                {'x': x, 'y': y, 'err': err, 'model': model})

            x, y, err = extract_metric(
                model_summary[param], 'time', 100 if model == 'NN' else 5000)
            time_data[param].append(
                {'x': x, 'y': y, 'err': err, 'model': model})

    plot_metric(jsd_data, type, 'log', 'jsd', 'JS-Divergence $D_{js}$')
    plot_metric(jsd_subsets_data, type, 'log',
                'subsets_jsd', 'JS-Divergence $D_{js}$')
    plot_metric(jsd_subsets_data, type, 'log',
                'whole_jsd', 'JS-Divergence $D_{js}$')
    plot_metric(w_jsd_data, type, 'log', 'w_jsd', 'JS-Divergence $D_{js}$')
    plot_metric(kt_data, type, 'linear', 'kt', "Kendall's $\\tau$")
    plot_metric(time_data, type, 'linear', 'time', "Training time in seconds$")


def sample_size_plot(summary_list, type):
    jsd_data = defaultdict(list)
    kt_data = defaultdict(list)
    param = 'M'
    model_summary = summary_list['NN']

    x, y, err = extract_metric(
        model_summary[param], 'jsd', no=5, x_val=lambda x: x['nsamples'])
    jsd_data[param].append(
        {'x': x, 'y': y, 'err': err, 'model': 'NN'})

    x, y, err = extract_metric(
        model_summary[param], 'kt', no=5, x_val=lambda x: x['nsamples'])
    kt_data[param].append(
        {'x': x, 'y': y, 'err': err, 'model': 'NN'})

    plot_metric(jsd_data, type, 'linear', 'jsd',
                'JS-Divergence $D_{js}$', 'samples', legend=False)


def test_new_sets(r, n=5):
    if('test_idx' in r['data']):
        data = {
            'pref_based': r['data']['pref_based'],
            'idx': r['data']['test_idx']
        }
        pred_freq = get_model_dist(r, data, 10000)
        actual_freq = np.array(r['data']['test_D'])
        return np.mean([jsd(d1, d2) for d1, d2 in zip(pred_freq, actual_freq)])

    i = 0
    new_set_idx = []
    idx = r['data']['idx']
    idx = idx if type(idx[0]) == list else [idx]
    idx = [tuple(i) for i in idx]
    while i < n:
        perm = np.random.permutation(range(len(r['M'])))[:3]
        if tuple(perm) not in idx:
            new_set_idx.append(perm)
            i += 1

    data = {
        'pref_based': r['data']['pref_based'],
        'idx': new_set_idx
    }
    pred_freq = get_model_dist(r, data, 1000)
    actual_freq = get_model_dist(r['data'], data, 10000)
    return np.mean([jsd(d1, d2) for d1, d2 in zip(pred_freq, actual_freq)])


def test_whole(r):
    data = {
        'pref_based': r['data']['pref_based'],
        'idx': [list(range(len(r['M'])))]
    }
    pred_freq = get_model_dist(r, data, 1000)
    actual_freq = r['whole_D'] if 'whole_D' in r else get_model_dist(r['data'], data, 10000) 
    return np.mean([jsd(d1, d2) for d1, d2 in zip(pred_freq, actual_freq)])


def summarize():
    style()
    for type in ['pref_based']:
        summary_list = {}
        for model in ['MLE', 'NN']:
            summary = {}
            summary_list[model] = summary
            baseDir = Path(f'results/{model}/{type}')
            for dir in baseDir.iterdir():
                if dir.is_dir():
                    param_name = dir.name
                    summary[param_name] = {}
                    for s in dir.iterdir():
                        if s.is_file() and s.name.endswith(".mat"):
                            set_name = s.name[:-4]
                            data = load_data(str(s))
                            set_evaluations(data, type)
                            subsets_jsd = np.array(
                                [test_new_sets(d) for d in data['results']])
                            whole_jsd = np.array(
                                [test_whole(d) for d in data['results']])
                            summary[param_name][set_name] = {
                                'no': data['results'][0]['no'],
                                'ncombs': data['results'][0]['ncombs'],
                                'nopts': data['results'][0]['nopts'],
                                'nsamples': data['nsamples'],
                                'mse': np.array([d['mse'] for d in data['results']]),
                                'jsd': np.array([d['jsd'] for d in data['results']]),
                                'subsets_jsd': subsets_jsd,
                                'whole_jsd': whole_jsd,
                                'kt': np.array([d['kt'] for d in data['results']]),
                                'w_jsd': np.array([d['w_jsd'] for d in data['results']]),
                                'kt1': np.array([d['kt1'] for d in data['results']]),
                                'kt2': np.array([d['kt2'] for d in data['results']]),
                                'time': np.array([d['time'] for d in data['results']]),
                            }
            print_summary(summary, model, type)
        option_size_plot(summary_list, type)
        if type == 'time_based':
            sample_size_plot(summary_list, type)


summarize()
