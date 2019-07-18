import functools
import operator
from time import time
from helpers.profiling import profile, global_profiler as profiler

import numpy as np
import torch
from torch.distributions import Bernoulli

from dft_net import DFT_Net

MAX_T = 5000


@profile
def train(dataset, opts):
    best_model = None
    best_error = 1e100
    ε = 1e-4
    it = 0
    delta_w = 0
    model = None
    start = time()
    no_progress_it = 0
    for it in range(opts['niter']):
        batch_loss, W_list = 0, []

        for j in range(len(dataset['M'])):
            if model is None:  # Initialize
                model = initialize(dataset, opts, j)
                print("Learning rate : {}".format(opts['lr']))
            else:
                update_fixed_parameters(dataset, model, opts, j)

            per_class_samples = get_per_class_samples(dataset, j, model, opts)
            predictions, W_list, avg_t, max_t = get_model_predictions(model, opts)

            pairs, confusion_matrix = align_samples(per_class_samples, predictions)
            batch_loss += compute_loss(opts, pairs, model)

        error = batch_loss.detach().numpy() / opts['ntrain']
        if error < best_error:
            best_error = error
            best_model = {
                "M": model.M.data.numpy().copy().tolist(),
                "φ1": float(model.φ1.data.numpy().copy()[0]),
                "φ2": float(model.φ2.data.numpy().copy()[0]),
                "w": model.w.copy().tolist(),
                "error": best_error,
                "iter": it + 1,
            }
            no_progress_it = 0

        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or error < ε:
            print_progress(best_model, error, it, model, opts, start)
            print([len(predictions[p]) / opts['ntrain'] for p in predictions])
            start = time()

        if error < ε:
            break

        profiler.start("calc_grad")
        if opts['w']:
            batch_loss.backward()
            W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list]).sum(axis=0) / len(W_list)
            delta_w = opts['momentum'] * delta_w + opts['lr'] / len(dataset['M']) * W_grad
            model.w -= delta_w.reshape(-1, 1)
            model.w = model.w.clip(0.1, 0.9)
            model.w /= model.w.sum()

        if opts['m']:
            opts['optimizer'].zero_grad()
            batch_loss.backward()
            opts['optimizer'].step()
            clamp_parameters(model, opts)
            model.update_S()

        profiler.finish("calc_grad")

        no_progress_it += 1

        if no_progress_it > 200:
            model = None
            no_progress_it = 0

    return best_model, it


@profile
def get_per_class_samples(dataset, j, model, opts):
    dist = dataset['freq'][j]
    n = opts['ntrain']
    # freq = np.random.multinomial(n, dist)

    freq = np.floor(np.array(dist) * n)
    while freq.sum() != n:
        d = freq / freq.sum()
        i = np.argmax(dist - d)
        freq[i] += 1

    per_class_samples = {x: freq[x] for x in range(model.options_count)}
    return per_class_samples


@profile
def print_progress(best_model, error, it, model, opts, start):
    print("." * 70)
    print("Iter {}/{}(time elapsed: {:0.3f}s)".format(it + 1, opts['niter'], time() - start))
    print("current err: {:0.5f}".format(error))
    print("best error: {:0.5f}".format(best_model["error"]))
    if opts['w']:
        print("current w: {}".format(model.w.T))
        print("best w: {}".format(best_model["w"]))


#    if opts['m']:
#     print("current M: \n{}".format(model.M.detach().numpy()))
#     print("best M: \n{}".format(np.array(best_model["M"])))


@profile
def compute_loss(opts, pairs, model):
    loss = 0

    for i in range(len(pairs)):
        target, p = pairs[i]
        n = p.view(1, -1)
        loss += opts['loss'](n, torch.tensor([target]))

    if opts['m']:  # L2 regularization
        large = model.M - torch.clamp(model.M, 0, 5)  # Discourage very big values
        large = large ** 2
        # row_sum = model.M.sum(dim=1)
        # small = torch.clamp(row_sum, 0.1) - row_sum  # Discourage all zero rows
        loss += 0.5 * large.sum()

    return loss


@profile
def get_model_predictions(model, opts):
    W_list, converged, t, avg_t = [], None, 0, 0
    predictions = {x: [] for x in range(model.options_count)}
    P = torch.Tensor(np.repeat(model.P0, opts['ntrain'], axis=1).tolist())
    n = opts['ntrain']

    while n > 0 and t < MAX_T:
        W = Bernoulli(probs=model.w[0][0]).sample([n])
        W = torch.stack([W, 1 - W])
        W.requires_grad = opts['w']
        W_list.append(W)

        P = model.forward(W, P)
        P_max, _ = P.max(0)
        idx = np.argwhere(P_max >= model.threshold).squeeze(dim=0)
        if converged is None or converged.shape[1] == 0:
            converged = P[:, idx]
        else:
            if len(idx) > 0:
                converged = torch.cat((converged, P[:, idx]), dim=1)

        avg_t += P.shape[1]
        idx = np.argwhere(P_max < model.threshold).squeeze(dim=0)
        P = P[:, idx]
        n = P.shape[1]
        t += 1

    if n > 0:
        if converged is None:
            converged = P
        else:
            converged = torch.cat((converged, P), dim=1)

    mx = converged.argmax(0)
    for i in range(model.options_count):
        predictions[i] = converged[:, np.argwhere(mx == i).squeeze(dim=0)]

    # convert to list of preferences per class
    predictions = {c: [predictions[c][:, i] for i in range(predictions[c].shape[1])] for c in predictions}

    return predictions, W_list, avg_t / opts['ntrain'], t


@profile
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


@profile
def initialize(data, opts, j):
    nn_opts = get_nn_options(data, opts, j)
    model = DFT_Net(nn_opts)
    loss, lr, momentum, optimizer = get_hyper_params(model, opts)
    opts['lr'] = lr
    opts['loss'] = loss
    opts['momentum'] = momentum
    opts['optimizer'] = optimizer
    return model


@profile
def reset_model(model, opts):
    loss, lr, momentum, optimizer = get_hyper_params(model, opts)
    opts['lr'] = lr
    opts['loss'] = loss
    opts['momentum'] = momentum
    opts['optimizer'] = optimizer


@profile
def align_samples(per_class_samples, preds):
    predictions = {c: preds[c].copy() for c in preds}
    per_class = {c: per_class_samples[c].copy() for c in per_class_samples}
    pairs = []
    options = len(per_class_samples)
    confusion_matrix = np.zeros((options, options))

    for cls in range(options):
        predictions[cls].sort(key=lambda p: p.detach().numpy().max())
        while per_class[cls] > 0 and len(predictions[cls]) > 0:
            per_class[cls] -= 1
            pred = predictions[cls].pop()
            pairs.append((cls, pred))
            confusion_matrix[cls, cls] += 1

    remaining = list(predictions.values())
    remaining = functools.reduce(operator.iconcat, remaining, [])
    remaining_cls = [c for c in per_class.keys() if per_class[c] > 0]
    remaining.sort(key=lambda x: x[remaining_cls].max())

    while len(remaining) > 0:
        pred = remaining.pop()
        p = pred.detach().numpy()
        label = np.argmax(p)
        cls = remaining_cls[np.argmax(p[remaining_cls])]
        if len(remaining) > 0 and per_class[cls] > 0:
            per_class[cls] -= 1
            pairs.append((cls, pred))
            confusion_matrix[cls, label] += 1
        if per_class[cls] == 0:
            remaining_cls.remove(cls)

    return pairs, confusion_matrix


@profile
def clamp_parameters(model, opts):
    if opts['m']:
        model.M.data.clamp_(min=0)


@profile
def get_hyper_params(model, opts):
    loss_func = torch.nn.MultiMarginLoss(margin=1)
    learning_rate = 0.001 if opts['w'] else 0.05
    momentum = 0.5
    if opts['m']:
        optim = torch.optim.SGD([model.M], lr=learning_rate, momentum=momentum)
    else:
        optim = None  # torch.optim.RMSprop([model.M], lr=learning_rate)
    return loss_func, learning_rate, momentum, optim


@profile
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
        'σ2': data['sigma2'],
        'threshold': data['threshold']
    }

    return options
