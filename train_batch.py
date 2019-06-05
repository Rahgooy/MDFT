"""
Learn MDFT using a recurrent neural network.
A dataset of different MDFT models is used to estimate common parameters used in those models.

Usage:
    train_batch.py [options]

Examples:
    train_batch.py --niter=100

Options:
    -h --help                  Show this screen.
    --niter=INT                Number of iterations. [default: 2000]
    --nprint=INT               Number of iterations per print. [default: 10]
    --ntest=INT                Number of test samples for evaluations[default: 500]
    --ntrain=INT               Number of train samples. [default: 500]
    --i=STR                    input data set. [default: data/threshold_o_3.json]
    --o=STR                    output path. [default: results/learn_m_pref_origw_single.txt]
    --m                        Learn M. [default: False]
    --w                        Learn W. [default: True]
    --s                        Learn S. [default: False]
"""
import functools
import json
import operator
from random import shuffle
from time import time

import torch
from scipy import stats
from dft import DFT
from dft_net import DFT_Net
from helpers.distances import hotaling_S
from helpers.weight_generator import RouletteWheelGenerator
from docpie import docpie
import numpy as np
from pprint import pprint
from pathlib import Path
from helpers.evaluation import dft_kl


def train(dataset, opts):
    best_model = None
    best_error = 1e100
    MAX_T = 100
    ε = 1e-4

    it = 0
    delta_w = 0
    model = None
    for it in range(opts['niter']):
        start = time()
        batch_loss = 0
        total_error = 0
        W_list = []
        for data in dataset:
            samples = opts['ntrain'] // len(dataset)
            if model is None:  # Initialize
                model = initialize(data, opts)
                print("Learning rate : {}".format(opts['lr']))
            else:
                update_fixed_parameters(data, model, opts)
            freqs = np.random.multinomial(samples, data['freq'])
            per_class_samples = {x: freqs[x] for x in range(model.options_count)}
            # per_class_samples = {x: int(data['freq'][x] * samples) for x in range(model.options_count)}
            # total = sum(per_class_samples.values())
            # if total < samples:
            #     f = np.array(data['freq'])
            #     diff = f * samples - np.round(f * samples)
            #     x = diff.argmax()
            #     per_class_samples[x] += 1

            predictions = {x: [] for x in range(model.options_count)}
            P = torch.Tensor(np.repeat(model.P0, samples, axis=1).tolist())
            gen = RouletteWheelGenerator(model.w)
            s = samples
            t = 0
            threshold = 0.7  # data['threshold']
            converged = []
            while s > 0 and t < MAX_T:
                W = np.array([gen.generate() for _ in range(s)]).squeeze().T
                if W.ndim == 1:
                    W = W.reshape(-1, s)
                W = torch.Tensor(W.tolist())
                W.requires_grad = opts['w']
                W_list.append(W)
                P = model.forward(W, P)

                P_min, _ = P.min(0)
                P_min = P_min.detach().numpy()
                P_max, _ = P.max(0)
                P_max = P_max.detach().numpy()
                P_max -= P_min
                P_sum = (P.detach().numpy() - P_min).sum(axis=0)

                P_max = P_max / P_sum
                s = 0
                for i in range(P.shape[1]):
                    if P_max[i] >= threshold:
                        converged.append(P[:, i])
                    else:
                        s += 1
                idx = np.argwhere(P_max < threshold).squeeze(axis=1)
                P = P[:, idx]
                t += 1

            for i in range(s):
                converged.append(P[:, i])

            for i in range(samples):
                pred = np.argmax(converged[i].detach().numpy()).tolist()
                predictions[pred].append(converged[i])

            pairs, confusion_matrix = align_samples(per_class_samples, predictions)

            for i in range(len(pairs)):
                target, p = pairs[i]
                s = p.view(1, -1)
                batch_loss += opts['loss'](s, torch.tensor([target]))

            total_error += batch_loss.detach().numpy()
            if opts['m']:
                opts['optimizer'].zero_grad()

            batch_loss.backward(retain_graph=True)

            if opts['w']:
                W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list])
                W_grad = W_grad.sum(axis=0)
                W_grad /= len(W_list)
                delta_w = opts['momentum'] * delta_w + opts['lr'] * 1 * W_grad
                model.w -= delta_w.reshape(-1, 1)
                model.w = model.w.clip(0.1, 0.9)
                model.w /= model.w.sum()

            if opts['m']:
                optimizer.step()
                clamp_parameters(model, opts)

        total_error /= opts['ntrain']
        if total_error < best_error:
            best_error = total_error
            best_model = {
                "M": model.M.data.numpy().copy().tolist(),
                "φ1": float(model.φ1.data.numpy().copy()[0]),
                "φ2": float(model.φ2.data.numpy().copy()[0]),
                "w": model.w.copy().tolist(),
                "error": best_error,
                "iter": it + 1
            }
        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or total_error < ε:
            print("." * 70)
            print("Iter {}/{}(time per iter: {:0.3f}s)".format(it + 1, opts['niter'], time() - start))
            print("w: {}".format(model.w.T))
            print("best: {}".format(best_model["error"]))
            print("w: {}".format(best_model["w"]))

        if total_error < ε:
            break

    return best_model, it


def update_fixed_parameters(data, model, opts):
    if not opts['w']:
        model.w = np.array(data['w'])
    if not opts['m']:
        model.M = torch.tensor(data['M'], requires_grad=False)
    if not opts['s']:
        model.b = torch.tensor([data['b']], requires_grad=False)
        model.φ1 = torch.tensor([data['φ1']], requires_grad=False)
        model.φ2 = torch.tensor([data['φ2']], requires_grad=False)


def initialize(data, opts):
    nn_opts = get_nn_options(data, opts)
    model = DFT_Net(nn_opts)
    loss, lr, momentum, optimizer = get_hyper_params(model, opts)
    opts['lr'] = lr
    opts['loss'] = loss
    opts['momentum'] = momentum
    opts['optimizer'] = optimizer
    return model


def align_samples(per_class_samples, pred):
    predictions = pred.copy()
    per_class = per_class_samples.copy()
    pairs = []
    options = len(per_class_samples)
    confusion_matrix = np.zeros((options, options))
    for cls in np.random.permutation(options):
        predictions[cls].sort(key=lambda p: p.detach().numpy().max())
        while per_class[cls] > 0 and len(predictions[cls]) > 0:
            per_class[cls] -= 1
            a = cls
            pred = predictions[cls].pop()
            pairs.append((a, pred))
            confusion_matrix[cls, cls] += 1

    remaining = list(predictions.values())
    remaining = functools.reduce(operator.iconcat, remaining, [])
    shuffle(remaining)
    while len(remaining) > 0:
        for cls in np.random.permutation(options):
            if len(remaining) > 0 and per_class[cls] > 0:
                remaining.sort(key=lambda p: p[cls])
                per_class[cls] -= 1
                a = cls
                pred = remaining.pop()
                pairs.append((a, pred))
                label = pred.detach().numpy().argmax()
                confusion_matrix[cls, label] += 1

    return pairs, confusion_matrix


def clamp_parameters(model, opts):
    if opts['m']:
        model.M.data.clamp_(min=0)


def get_hyper_params(model, opts):
    loss_func = torch.nn.MultiMarginLoss(margin=1e-2)
    learning_rate = 0.005
    momentum = 0.5
    if opts['m']:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    else:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    return loss_func, learning_rate, momentum, optim


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


def check_data(dataset, opts):
    params = {'w': 'w', 'M': 'm', 'φ1': 's', 'φ2': 's', 'b': 's'}
    passed = True
    for p in params:
        if opts[params[p]]:
            w = dataset[0][p]
            for d in dataset:
                if d[p] != w:
                    print(f"Dataset check failed: cannot learn {p} when data samples are "
                          f"generated with different {p}.")
                    passed = False
                    break
    print("Dataset check passed.")
    return passed


def get_nn_options(data, opts):
    M = torch.tensor(data['M'], requires_grad=False)
    o, a = M.shape[0], M.shape[1]
    if opts['m']:
        M = torch.rand((o, a), requires_grad=True)
        M.data.clamp_(min=0)

    if opts['w']:
        w = np.ones((a, 1))
        w /= w.sum()
    else:
        w = np.array(data['w'])
    options = {
        'φ1': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['φ1']], requires_grad=False),
        'φ2': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['φ2']], requires_grad=False),
        'b': torch.tensor([10.0], requires_grad=True) if opts['s']
        else torch.tensor([float(data['b'])], requires_grad=False),
        'options_count': o,
        'attr_count': a,
        'M': M,
        'P0': np.array(data['P0']),
        'w': w,
    }

    return options


def main():
    """Command line argument processing"""
    main_start = time()
    opts = get_options()
    pprint(opts)

    with open(opts['i'], 'r', encoding="UTF-8") as f:
        data = json.load(f)
    datasets = data['datasets']

    if not check_data(datasets, opts):
        return

    best, it = train(datasets[:5], opts)
    return
    outPath = Path(output)
    outPath.parent.mkdir(exist_ok=True, parents=True)
    with outPath.open(mode='w') as f:
        data.summary(f)
        print("============================= Settings ==============================", file=f)
        print("Repetition : {}".format(rep + 1), file=f)
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
