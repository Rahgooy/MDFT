import functools
import operator
from random import shuffle
from time import time

import numpy as np
import torch
from torch.distributions import Bernoulli

from dft_net import DFT_Net


def train(dataset, opts):
    best_model = None
    best_error = 1e100
    MAX_T = 1000
    ε = 1e-4
    it = 0
    delta_w = 0
    model = None

    for it in range(opts['niter']):
        start = time()
        batch_loss = 0
        total_error = 0
        W_list = []
        for j in range(len(dataset['M'])):
            samples = opts['ntrain']
            if model is None:  # Initialize
                model = initialize(dataset, opts, j)
                print("Learning rate : {}".format(opts['lr']))
            else:
                update_fixed_parameters(dataset, model, opts, j)
            freqs = np.random.multinomial(samples, dataset['freq'][j])
            per_class_samples = {x: freqs[x] for x in range(model.options_count)}

            predictions = {x: [] for x in range(model.options_count)}
            P = torch.Tensor(np.repeat(model.P0, samples, axis=1).tolist())
            n = samples
            t = 0
            threshold = dataset['threshold']
            converged = None
            while n > 0 and t < MAX_T:
                W = Bernoulli(probs=model.w[0][0]).sample([n])
                W = torch.stack([W, 1 - W])

                W.requires_grad = opts['w']
                W_list.append(W)
                P = model.forward(W, P)

                P_max, _ = P.max(0)
                idx = np.argwhere(P_max >= threshold).squeeze(dim=0)
                if converged is None or converged.shape[1] == 0:
                    converged = P[:, idx]
                else:
                    if len(idx) > 0:
                        converged = torch.cat((converged, P[:, idx]), dim=1)

                idx = np.argwhere(P_max < threshold).squeeze(dim=0)
                P = P[:, idx]
                n = P.shape[1]
                t += 1

                # n = 0
                # bigger = P_max >=
                # for i in range(P.shape[1]):
                #     if P_max[i] >= threshold:
                #         converged.append(P[:, i])
                #     else:
                #         n += 1
                # idx = np.argwhere(P_max < threshold).squeeze(axis=1)
                # P = P[:, idx]
                # t += 1
            if n > 0:
                if converged is None:
                    converged = P
                else:
                    converged = torch.cat((converged, P), dim=1)

            for i in range(samples):
                _, pred = converged[:, i].max(0)
                pred = pred.detach().data.tolist()
                predictions[pred].append(converged[:, i])

            pairs, confusion_matrix = align_samples(per_class_samples, predictions)

            for i in range(len(pairs)):
                target, p = pairs[i]
                n = p.view(1, -1)
                batch_loss += opts['loss'](n, torch.tensor([target]))

            total_error += batch_loss.detach().numpy()
            if opts['m']:
                opts['optimizer'].zero_grad()

            batch_loss.backward(retain_graph=True)

            if opts['w']:
                W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list])
                W_grad = W_grad.sum(axis=0)
                W_grad /= len(W_list)
                delta_w = opts['momentum'] * delta_w + opts['lr'] / len(dataset['M']) * W_grad
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
            print("err: {}".format(total_error))
            print("w: {}".format(model.w.T))
            print("best: {}".format(best_model["error"]))
            print("w: {}".format(best_model["w"]))

        if total_error < ε:
            break

    return best_model, it


def update_fixed_parameters(data, model, opts, j):
    if not opts['w']:
        model.w = np.array(data['w'])
    if not opts['m']:
        model.M = torch.tensor(data['M'][j], requires_grad=False, dtype=torch.float)
    if not opts['s']:
        model.b = torch.tensor([float(data['b'])], requires_grad=False)
        model.φ1 = torch.tensor([float(data['φ1'])], requires_grad=False)
        model.φ2 = torch.tensor([float(data['φ2'])], requires_grad=False)
    model.update_S()


def initialize(data, opts, j):
    nn_opts = get_nn_options(data, opts, j)
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
    learning_rate = 0.001
    momentum = 0.5
    if opts['m']:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    else:
        optim = torch.optim.RMSprop([model.M], lr=learning_rate)
    return loss_func, learning_rate, momentum, optim


def get_nn_options(data, opts, j):
    M = torch.tensor(data['M'][j], requires_grad=False)
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
        'P0': np.zeros((o, 1)),
        'w': w,
        'σ2': data['sigma2']
    }

    return options