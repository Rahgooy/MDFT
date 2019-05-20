from pathlib import Path
import re
import numpy as np
import scipy.stats

from dft import load_DFT_dataset, get_fixed_T_dft_dist, DFT
from helpers.distances import hotaling_S
from helpers.evaluation import jsd, kendalltau_dist, dft_jsd
from matplotlib import pyplot as plt
import copy

print_details = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    x, y = scipy.stats.t.interval(confidence, n - 1, loc=m, scale=se)
    return m, se, m - x


def get_intersection(actual, predicted):
    return len(set(actual[:, 0]).intersection(predicted[:, 0])), \
           len(set(actual[:, 1]).intersection(predicted[:, 1])),


def get_attr_index(M, M_, non_zero):
    m = M[non_zero]
    m_ = M_[non_zero]

    kt1 = min(kendalltau_dist(m[:, 0], m_[:, 0]), kendalltau_dist(m[:, 1], m_[:, 1]))
    kt2 = min(kendalltau_dist(m[:, 0], m_[:, 1]), kendalltau_dist(m[:, 1], m_[:, 0]))
    if kt1 <= kt2:
        return [0, 1]
    return [1, 0]


def _get_result_from_text(name, text):
    dataset = re.findall(r"_(\d*).txt", name)[0]
    iterations = re.findall(r"Number of Iterations : (.*)\[", text)[0]
    rep = re.findall(r"Repetition : (.*)", text)[0]
    time = re.findall(r"Time : (.*)s", text)[0]
    n = re.findall(r"Test sample size: (.*)", text)[0]
    actual = re.findall(r"Actual freq: \[(.*)\]", text)[0]
    actual = [int(x) for x in actual.split('&')]
    actual_dist = np.array(actual)
    actual_dist = actual_dist / actual_dist.sum()

    model = re.findall(r"Model freq: \[(.*)\]", text)[0]
    model = [int(x) for x in model.split('&')]
    model_dist = np.array(model)
    model_dist = model_dist / model_dist.sum()

    kl = re.findall(r"KL-Divergence: \[(.*)\]", text)[0]

    Ms = re.findall(r"M:[\n|\r]*(\[\[[+-e|\[\].\s\d\n\r]*\]\])", text)

    M = Ms[0].replace('[', '').replace(']', '').replace('\n', '')
    M = np.fromstring(M, sep=' ').reshape(-1, 2)

    M_ = Ms[1].replace('[', '').replace(']', '').replace('\n', '')
    M_ = np.fromstring(M_, sep=' ').reshape(-1, 2)

    non_zero = actual_dist > 0
    idx = get_attr_index(M, M_, non_zero)

    kt1 = kendalltau_dist(M[non_zero, 0], M_[non_zero, idx[0]])
    kt2 = kendalltau_dist(M[non_zero, 1], M_[non_zero, idx[1]])
    return {
        'iterations': int(iterations),
        'time': float(time),
        'rep': float(rep),
        'test_samples': int(n),
        'actual_freq': actual,
        'model_freq': model,
        'jsd': jsd(actual_dist, model_dist),
        'kl': float(kl),
        'set': int(dataset),
        'kt1': kt1,
        'kt2': kt2,
        'M': M,
        'M_': M_
    }


def _get_result_from_json(f):
    dataset = re.findall(r"_(\d*).json", f.name)[0]
    with open(f) as file:
        import json
        r = json.load(file)

    M = r['data']['M']
    M = np.array(M)

    M_ = r['best']['M']
    M_ = np.array(M_)
    actual_dist = np.array(r['actual_dist']).squeeze()
    model_dist = np.array(r['model_dist']).squeeze()
    data_dist = np.array(r['data']['dist']).squeeze()
    non_zero = data_dist > 0
    idx = get_attr_index(M, M_, non_zero)
    re_order = idx[0]

    kt1 = kendalltau_dist(M[non_zero, 0], M_[non_zero, idx[0]])
    kt2 = kendalltau_dist(M[non_zero, 1], M_[non_zero, idx[1]])

    w = np.array(r['data']['w']).squeeze()
    w_ = np.array(r['best']['w']).squeeze()
    w = w[idx]
    w_ = w_[idx]

    return {
        'iterations': r['iter'],
        'time': r['time'],
        'rep': 1,
        'test_samples': r['ntest'],
        'actual_freq': [int(x) for x in r['actual_freq']],
        'model_freq': [int(x) for x in r['model_freq']],
        'jsd': jsd(actual_dist, model_dist),
        'kl': 0,
        'set': int(dataset),
        'kt1': kt1,
        'kt2': kt2,
        'M': M,
        'M_': M_,
        'w': w,
        'w_': w_,
        'w_jsd': jsd(w, w_),
        're_order': re_order
    }


def summarize(dir_path, pattern):
    dir = Path(dir_path)
    results = []
    jsd_list = []
    w_jsd_list = []
    time_list = []
    iter_list = []
    kt_list1 = []
    kt_list = []
    kt_list2 = []
    for f in dir.iterdir():
        if f.is_file() and f.match(pattern):
            # try:
            # r = _get_result_from_text(f.name, f.read_text())
            r = _get_result_from_json(f)
            # except:
            #    continue
            jsd_list.append(r["jsd"])
            w_jsd_list.append(r["w_jsd"])

            kt_list1.append(r['kt1'])
            kt_list2.append(r['kt2'])
            kt_list.append((r['kt1'] + r['kt2']) / 2)
            time_list.append(r['time'] / r['rep'])
            iter_list.append(r['iterations'])

            results.append(r)

    if len(kt_list1) > 0:
        m, se, h = mean_confidence_interval(jsd_list)
        jsd = {
            'm': m,
            'se': se,
            'h': h
        }
        m, se, h = mean_confidence_interval(w_jsd_list)
        w_jsd = {
            'm': m,
            'se': se,
            'h': h
        }
        m1, se1, h1 = mean_confidence_interval(kt_list1)
        m2, se2, h2 = mean_confidence_interval(kt_list2)
        m, se, h = mean_confidence_interval(kt_list)
        kt = {
            'm': m,
            'se': se,
            'h': h,
            'm1': m,
            'se1': se1,
            'h1': h1,
            'm2': m2,
            'se2': se2,
            'h2': h2,
        }
        time = sum(time_list) / len(time_list)
        iter = sum(iter_list) / len(iter_list)

    else:
        jsd = {
            'm': 0,
            'se': 0,
            'h': 0
        }
        w_jsd = jsd
        kt = {
            'm': 0,
            'se': 0,
            'h': 0,
            'm1': 0,
            'se1': 0,
            'h1': 0,
            'm2': 0,
            'se2': 0,
            'h2': 0,
        }
        time = 0
        iter = 0

    return sorted(results, key=lambda x: x['set']), jsd, w_jsd, kt, time, iter


def print_data(data, jsd, kt, details=True):
    if details:
        print("{:<10}{:<10} {:<15} {:<15} {:^10}{:^10}{:^10}   \t{:^20} \t{:^20} \t{:^10}"
              .format("set", "it", "time", "samples", "jsd", "kt1", "kt2", "actual_w", "model_w", "reorder"))
        for i, d in enumerate(data):
            print("{:<10}{:<10} {:<15.1f} {:<14} {:>10f}{:>10f}{:>10f}   \t{:^20s} \t{:^20s} \t{:^10}".format(
                d["set"],
                d["iterations"],
                d["time"],
                d["test_samples"],
                d["jsd"],
                d["kt1"],
                d["kt2"],
                ", ".join(["{:0.3f}".format(x) for x in d["w"]]),
                ", ".join(["{:0.3f}".format(x) for x in d["w_"]]),
                d["re_order"]
            ))
    print("JSD-mean: {:.4f} ± {:.4f}".format(jsd['m'], jsd['h']))
    print("JSD-se: {:.4f}".format(jsd['se']))
    print("95% CI: [{:.4f}, {:.4f}]".format(jsd['m'] - jsd['h'], jsd['m'] + jsd['h']))
    print("")

    print("KT1-mean: {:.4f} ± {:.4f}".format(kt['m1'], kt['h1']))
    print("se: {:.4f}".format(kt['se1']))
    print("95% CI: [{:.4f}, {:.4f}]".format(kt['m1'] - kt['h1'], kt['m1'] + kt['h1']))
    print("")

    print("KT2-mean: {:.4f} ± {:.4f}".format(kt['m2'], kt['h2']))
    print("se: {:.4f}".format(kt['se2']))
    print("95% CI: [{:.4f}, {:.4f}]".format(kt['m2'] - kt['h2'], kt['m2'] + kt['h2']))
    print("")


def analyze_sample_size(dir, save=False):
    sample_sizes = ['20', '30', '50', '100', '150']
    options = [5, 7, 9]

    jsd_list = {i: {j: None for j in options} for i in sample_sizes}
    w_jsd_list = {i: {j: None for j in options} for i in sample_sizes}
    kt_list = {i: {j: None for j in options} for i in sample_sizes}
    time_list = {i: {j: None for j in options} for i in sample_sizes}
    iter_list = {i: {j: None for j in options} for i in sample_sizes}
    for s_size in sample_sizes:
        for o in options:
            p = f"results/{dir}/random/set_o_{o}/"
            if not Path(p).exists():
                continue
            # try:
            data, jsd, w_jsd, kt, time, iter = summarize(p, f"n_{s_size}_*json")
            # except:
            #     continue

            jsd_list[s_size][o] = jsd
            w_jsd_list[s_size][o] = w_jsd
            kt_list[s_size][o] = kt
            time_list[s_size][o] = time
            iter_list[s_size][o] = iter

            print("-" * 175)
            print(f"options: {o}")
            print(f"sample size: {s_size}")
            print_data(data, jsd, kt, print_details)

    width = 25
    x = np.arange(len(sample_sizes)) * 100
    for i, o in enumerate(options):
        jsd_points = [0 if jsd_list[s][o] is None else jsd_list[s][o]['m'] for s in sample_sizes]
        jsd_errors = [0 if jsd_list[s][o] is None else jsd_list[s][o]['h'] for s in sample_sizes]
        plt.bar(x + width * i, jsd_points, yerr=jsd_errors, width=width, capsize=2, label=f'{o} options')

    plt.xticks(x + 25, sample_sizes)
    plt.ylabel('$D_{js}$')
    plt.xlabel('Sample size')
    plt.title('Average $D_{js}$ with 95% CI', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{dir}/divergence.pdf')
    else:
        plt.show()
    plt.close()

    for i, o in enumerate(options):
        jsd_points = [0 if w_jsd_list[s][o] is None else w_jsd_list[s][o]['m'] for s in sample_sizes]
        jsd_errors = [0 if w_jsd_list[s][o] is None else w_jsd_list[s][o]['h'] for s in sample_sizes]
        plt.bar(x + width * i, jsd_points, yerr=jsd_errors, width=width, capsize=2, label=f'{o} options')

    plt.xticks(x + 25, sample_sizes)
    plt.ylabel('$D_{js}$')
    plt.xlabel('Sample size')
    plt.title('Average $D_{js}$ of attention distributions with 95% CI', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{dir}/W_divergence.pdf')
    else:
        plt.show()
    plt.close()

    if dir.startswith("learn_m"):
        for i, o in enumerate(options):
            points = [0 if kt_list[s][o] is None else kt_list[s][o]['m'] for s in sample_sizes]
            errors = [0 if kt_list[s][o] is None else kt_list[s][o]['h'] for s in sample_sizes]
            plt.bar(x + width * i, points, yerr=errors, width=width, capsize=2, label=f'{o} options')

        plt.xticks(x + 25, sample_sizes)
        plt.ylabel('$\\tau$')
        plt.xlabel('Sample size')
        plt.title('Average Kendall\'s $\\tau$ with 95% CI', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'results/{dir}/tau.pdf')
        else:
            plt.show()
        plt.close()

        s = '100'
        points = [0 if kt_list[s][o] is None else kt_list[s][o]['m'] for o in options]
        errors = [0 if kt_list[s][o] is None else kt_list[s][o]['h'] for o in options]
        plt.bar(np.arange(len(options)) * 40, [points[0], 0, 0], yerr=[errors[0], 0, 0], width=20, capsize=4)
        plt.bar(np.arange(len(options)) * 40, [0, points[1], 0], yerr=[0, errors[1], 0], width=20, capsize=4)
        plt.bar(np.arange(len(options)) * 40, [0, 0, points[2]], yerr=[0, 0, errors[2]], width=20, capsize=4)

        plt.xticks(np.arange(len(options)) * 40, options)
        plt.ylabel('$\\tau$')
        plt.xlabel('Number of options')
        plt.title('Average Kendall\'s $\\tau$ with 95% CI', fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(f'results/{dir}/tau2_{s}.pdf')
        else:
            plt.show()
        plt.close()

    for i, o in enumerate(options):
        points = [time_list[s][o] for s in sample_sizes]
        plt.bar(x + width * i, points, width=width, label=f'{o} options')

    plt.xticks(x + 25, sample_sizes)
    plt.ylabel('$time(s)$')
    plt.xlabel('Sample size')
    plt.title('Average training time in seconds', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{dir}/time.pdf')
    else:
        plt.show()
    plt.close()


def analyze_options_count():
    for i in range(2, 7):
        o = 5 + (i - 3) * 2
        print(o)
        p = f"results/learn_m/random/set_{i}/"
        if not Path(p).exists():
            continue
        data, j, kt = summarize(p, "n_100_*")
        print_data(data, j, kt, print_details)


def analyze_w():
    set = 3
    sample = 2
    o = 5
    for sample in range(1, 101):
        if not Path(f"results/learn_m/random/set_{set}/", f"n_100_l100_o{o}_{sample}.txt").exists():
            break
        results = summarize(f"results/learn_m/random/set_{set}/", f"n_100_l100_o{o}_{sample}.txt")
        M_ = results[0][0]['M_']
        data = load_DFT_dataset(f"data/random/set_{set}/set_hotaling_n100_l100_o{o}_{sample}.pickle")
        W = data.w
        w0 = W[0][0]
        φ1, φ2, b = data.parameters['φ1'], data.parameters['φ2'], data.parameters['b']

        S = hotaling_S(M_, φ1, φ2, b)
        data_ = DFT(M_, S, W.copy(), data.P0)
        jsd0, a, e = dft_jsd(data, data_, 1000, data.samples[0].t)
        jsd0 = jsd0[0]

        jsd_list = []
        w_list = []
        for w in np.linspace(0, 1, 100):
            data_.w[0] = data.w[0] = w
            data_.w[1] = data.w[1] = 1 - w
            jsd, _, _ = dft_jsd(data, data_, 250, data.samples[0].t)
            jsd_list.append(jsd)
            w_list.append(w)
        plt.plot(w_list, jsd_list, label="W")
        plt.scatter(w0, jsd0, marker="*", color='r', label="True W")
        plt.xlabel("$W_1$")
        plt.ylabel("Divergence")
        plt.legend()
        plt.savefig(f"results/learn_m/W_analysis/W2_{sample}.png")
        plt.close()


# analyze_w()
# analyze_options_count()
analyze_sample_size("learn_w", True)
