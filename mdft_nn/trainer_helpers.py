import functools
import operator

from mdft_nn.mdft import MDFT, get_preference_based_dft_dist, get_time_based_dft_dist
from mdft_nn.helpers.distances import hotaling_S
from mdft_nn.helpers.profiling import profile
from mdft_nn.mdft_net import MDFT_Net

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli, Uniform


MAX_T = 200


@profile
def get_per_class_samples(model, dist, n):
    freq = np.floor(np.array(dist) * n)
    while freq.sum() != n:
        d = freq / freq.sum()
        i = np.argmax(dist - d)
        freq[i] += 1

    per_class_samples = {x: freq[x] for x in range(model.options_count)}
    return per_class_samples


@profile
def print_progress(best_model, error, mse, it, opts, time_span):
    if it == 0:
        print("." * 90)
        print(f"{'Iteration':16s} {'time':10s} {'curr err':<10s} {'curr mse':10s} {'best mse':10s} "
              f"{'avg delib':15s} {'predicted w0'}")
    print(f"{it + 1:5d}/{opts['niter']:<10d} {time_span:<10.3f} {error:<10.3f} {mse:<10.3f} {best_model['mse']:<10.4f} "
          f"{best_model['avg_t']:<15.2f} "
          f"{best_model['w'][0][0]:0.3f}")


@profile
def compute_loss(pairs, nn_opts):
    loss = 0

    for i in range(len(pairs)):
        target, p = pairs[i]
        n = p.view(1, -1)
        loss += nn_opts['loss'](n, torch.tensor([target]))

    return loss


@profile
def get_model_predictions(model, learn_w, nsamples, pref_based):
    converged, t, avg_t = None, 0, 0
    predictions = {x: [] for x in range(model.options_count)}
    P = torch.Tensor(np.repeat(model.P0, nsamples, axis=1).tolist())

    if pref_based:
        n = nsamples
        while n > 0 and t < MAX_T:
            W = F.gumbel_softmax(logits=model.w.T.repeat((n, 1)), hard=True).T
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
    else:
        for i in range(model.threshold):
            W = F.gumbel_softmax(logits=model.w.T.repeat(
                (nsamples, 1)), hard=True).T
            P = model.forward(W, P)
        converged = P
        avg_t = model.threshold * nsamples

    mx = converged.argmax(0)
    for i in range(model.options_count):
        predictions[i] = converged[:, np.argwhere(mx == i).squeeze(dim=0)]

    # convert to list of preferences per class
    predictions = {c: [predictions[c][:, i]
                       for i in range(predictions[c].shape[1])] for c in predictions}

    return predictions, avg_t / nsamples, t


@profile
def get_nn_model(nn_opts, idx):
    o = nn_opts.copy()
    o['M'] = o['M'][idx]
    o['P0'] = np.zeros((len(idx), 1))
    o['options_count'] = len(idx)
    model = MDFT_Net(o)
    return model


@profile
def initi_nn_opts(opts, data):
    nn_opts = get_nn_options(data, opts)
    loss, lr, optimizer, grad_clip = get_hyper_params(
        nn_opts, opts)
    nn_opts['lr'] = lr
    nn_opts['loss'] = loss
    nn_opts['optimizer'] = optimizer
    nn_opts['grad_clip'] = grad_clip
    return nn_opts


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
        per_class[cls] -= 1
        pairs.append((cls, pred))
        confusion_matrix[cls, label] += 1
        if per_class[cls] == 0:
            remaining_cls.remove(cls)

    return pairs, confusion_matrix


@profile
def clamp_parameters(nn_opts, opts):
    if opts['m']:
        nn_opts['M'].data.clamp_(min=0)


@profile
def get_hyper_params(nn_opts, opts):
    loss_func = torch.nn.MultiMarginLoss(margin=1e0)
    lr = 0.05
    grad_clip = 50

    if 'grad_clip' in opts:
        grad_clip = opts['grad_clip']
    if 'lr' in opts:
        lr = opts['lr']
    optim = torch.optim.Adam([nn_opts['M'], nn_opts['w']], lr=lr)
    return loss_func, lr, optim, grad_clip


@profile
def get_nn_options(data, opts):
    M = torch.tensor(data['M'], requires_grad=False)
    o, a = len(data['idx'][0]), M.shape[1]
    if opts['m']:
        u = Uniform(1, 10)
        M = u.sample(M.shape)
        M.requires_grad = True
        M.data.clamp_(min=0)

    if opts['w']:
        w = np.ones((a, 1))
        w = torch.tensor(w.tolist(), requires_grad=True)
    else:
        w = torch.tensor(data['w'], requires_grad=False)
    options = {
        'phi1': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['phi1']], requires_grad=False),
        'phi2': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['phi2']], requires_grad=False),
        'b': torch.tensor([10.0], requires_grad=True) if opts['s']
        else torch.tensor([float(data['b'])], requires_grad=False),
        # 'options_count': o,
        'attr_count': a,
        'M': M,
        # 'P0': np.zeros((o, 1)),
        'w': w,
        'sig2': data['sig2'],
        'threshold': data['threshold']
    }

    return options


@profile
def get_model_dist(model, data, n):
    freq_list = []
    for idx in data['idx']:
        M = np.array(model['M'])[idx]
        S = hotaling_S(M, model['phi1'], model['phi2'], model['b'])
        P0 = np.zeros((M.shape[0], 1))
        m = MDFT(M, S, np.array(model['w']), P0, np.array(model['sig2']))
        if data['pref_based']:
            f = get_preference_based_dft_dist(m, n, model["threshold"])
        else:
            f = get_time_based_dft_dist(m, n, model["threshold"])

        freq_list.append(f.squeeze().tolist())
    return freq_list
