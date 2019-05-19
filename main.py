"""
Learn DFT using a recurrent neural network

Usage:
    main.py [options]

Examples:
    main.py --niter=100

Options:
    -h --help                  Show this screen.
    --niter=INT                Number of iterations. [default: 2]
    --nprint=INT               Number of iterations per print. [default: 10]
    --ntest=INT                Number of test samples for evaluations[default: 500]
    --i=STR                    input data set. [default: data/random/set_o_5/set_hotaling_n100_l100_o5_2.pickle]
    --o=STR                    output path. [default: results/learn_m_pref_origw_single.txt]
    --m                        Learn M. [default: False]
    --w                        Learn W. [default: False]

"""
import functools
import json
import operator
from random import shuffle
from time import time

import torch
from scipy import stats
from dft import load_DFT_dataset, DFT, get_dft_dist
from dft_net import DFT_Net
from helpers.distances import hotaling_S
from helpers.weight_generator import RouletteWheelGenerator
from docpie import docpie
import numpy as np
from pprint import pprint
from pathlib import Path
from helpers.evaluation import dft_kl


def train():
    print("Learning rate : {}".format(lr))
    best_model = None
    best_error = 1e100
    ε = 1e-4
    samples = data.samples.copy()
    clamp_parameters()

    it = 0
    targets = [np.argmax(d.choice).tolist() for d in samples]
    per_class_samples = {x: 0 for x in range(model.options_count)}
    zero_ops = np.ones(model.M.shape[0])
    for t in targets:
        per_class_samples[t] += 1
        zero_ops[t] = 0
    delta_w = 0
    for it in range(niter):
        start = time()
        total_error = 0
        predictions = {x: [] for x in range(model.options_count)}
        P = torch.Tensor(np.repeat(model.P0, len(samples), axis=1).tolist())
        W_list = []
        gen = RouletteWheelGenerator(model.w)
        for t in range(1, T + 1):
            W = np.array([gen.generate() for _ in range(len(samples))]).squeeze().T
            W = torch.Tensor(W.tolist())
            W.requires_grad = learn_w
            W_list.append(W)
            P = model.forward(W, P)

        for i in range(len(targets)):
            pred = np.argmax(P[:, i].detach().numpy()).tolist()
            predictions[pred].append(P[:, i])

        pairs, confusion_matrix = align_samples(per_class_samples, predictions)

        batch_loss = 0
        for i in range(len(pairs)):
            target, p = pairs[i]
            s = p.view(1, -1)
            batch_loss += loss(s, torch.tensor([target]))

        total_error += batch_loss.detach().numpy()
        total_error /= len(data.samples)

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
        rows = zero_ops == 0
        if it % nprint == 0 or it == niter - 1 or total_error < ε:
            print("." * 70)
            print("Iter {}/{}(time per iter: {:0.3f}s)".format(it + 1, niter, time() - start))
            print("{}[{}]: {}".format(loss_name, optim_name, total_error))
            print("w: {}".format(model.w.T))
            print("best: {}".format(best_model["error"]))
            print("w: {}".format(best_model["w"]))

            # print("Confusion matrix:")
            # print(confusion_matrix)
            # print("M")

            # print(data.M[rows])
            # print("")
            # print(model.M.data.numpy()[rows])
            # print("best M")
            # print(np.array(best_model["M"]))

        if total_error < ε:
            break
        if learn_m:
            optimizer.zero_grad()

        batch_loss.backward(retain_graph=True)
        # if it % nprint == 0 or it == niter - 1:
        #     print("Gradients: ")
        #     print(-model.M.grad.detach().numpy()[rows])

        if learn_w:
            W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list])
            W_grad = W_grad.sum(axis=0)
            W_grad /= T
            delta_w = momentum * delta_w + lr * 1 * W_grad
            model.w -= delta_w.reshape(-1, 1)
            model.w = model.w.clip(0.1, 0.9)
            model.w /= model.w.sum()

        if learn_m:
            optimizer.step()
            clamp_parameters()

    return best_model, it


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


def clamp_parameters():
    if learn_m:
        model.M.data.clamp_(min=0)


def get_hyper_params():
    loss_func = torch.nn.MultiMarginLoss(margin=1e-2)
    learning_rate = 0.005
    momentum = 0.5
    if learn_m:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    else:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    return loss_func, learning_rate, momentum, optim


if __name__ == "__main__":
    """Command line argument processing"""
    main_start = time()
    opts = docpie(__doc__)

    # remove '--' in front of option names
    opts = {key.replace('--', ''): item for key, item in opts.items()}
    pprint(opts)
    niter = int(opts['niter'])
    ntest = int(opts['ntest'])
    nprint = int(opts['nprint'])
    output = opts['o']
    learn_m = eval(str(opts['m']))
    learn_w = eval(str(opts['w']))

    data = load_DFT_dataset(opts['i'])
    data.summary()

    options = {
        'b': data.parameters['b'],
        'options_count': data.M.shape[0],
        'attr_count': data.M.shape[1],
        'M': data.M.copy(),
        'φ1': data.parameters['φ1'],
        'φ2': data.parameters['φ2'],
        'P0': data.P0.copy(),
        'w': None if learn_w else data.w.copy(),
        'learn_m': learn_m,
        'learn_w': learn_w
    }

    T = data.samples[0].t
    best = None
    it = None
    for rep in range(1):
        model = DFT_Net(options)
        loss, lr, momentum, optimizer = get_hyper_params()
        optim_name = str(optimizer).split('(')[0]
        loss_name = str(loss).split('(')[0]
        b, i = train()
        if best is None or b['error'] < best['error']:
            best = b
            it = i
        if b['error'] < 1e-10:
            break

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
