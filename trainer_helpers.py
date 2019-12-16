import functools
import operator

from dft import DFT, get_threshold_based_dft_dist
from helpers.distances import hotaling_S
from helpers.profiling import profile

import numpy as np
import torch
from torch.distributions import Bernoulli, Uniform

from dft_net import DFT_Net

MAX_T = 5000


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
              f"{'avg delib':10s} {'predicted w0':15s} {'M params':20s}")
    print(f"{it + 1:5d}/{opts['niter']:<10d} {time_span:<10.3f} {error:<10.3f} {mse:<10.3f} {best_model['mse']:<10.4f} "
          f"{best_model['avg_t']:<10.2f} {best_model['w'][0][0]:<15.3f} {best_model['M_params']}")


@profile
def compute_loss(pairs, nn_opts):
    loss = 0

    for i in range(len(pairs)):
        target, p = pairs[i]
        n = p.view(1, -1)
        loss += nn_opts['loss'](n, torch.tensor([target]))

    return loss


@profile
def get_model_predictions(model, learn_w, nsamples):
    W_list, converged, t, avg_t = [], None, 0, 0
    predictions = {x: [] for x in range(model.options_count)}
    P = torch.Tensor(np.repeat(model.P0, nsamples, axis=1).tolist())
    n = nsamples

    while n > 0 and t < MAX_T:
        W = Bernoulli(probs=model.w[0][0]).sample([n])
        W = torch.stack([W, 1 - W])
        W.requires_grad = learn_w
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

    return predictions, W_list, avg_t / nsamples, t


def normalize_m(M):
    M[:, 0] = M[:, 0] / M[:, 0].sum()
    M[:, 1] = M[:, 1] / M[:, 1].sum()
    return M


@profile
def get_nn_model(nn_opts, idx):
    o = nn_opts.copy()
    o['M'] = torch.zeros(o['M'].shape, requires_grad=False)

    if o['parametric_m']:
        o['M'][:, 0] = get_parametric_attr_values(nn_opts, 0)
        o['M'][:, 1] = get_parametric_attr_values(nn_opts, 1)
    else:
        if o['normalize']:
            o['M'][:, 0] = nn_opts['M'][:, 0] / nn_opts['M'][:, 0].sum()
            o['M'][:, 1] = nn_opts['M'][:, 1] / nn_opts['M'][:, 1].sum()

    o['M'] = o['M'][idx]
    model = DFT_Net(o)
    return model


@profile
def initi_nn_opts(opts, data):
    nn_opts = get_nn_options(data, opts)
    loss, w_lr, m_lr, w_decay, momentum, optimizer = get_hyper_params(nn_opts, opts)
    nn_opts['w_lr'] = w_lr
    nn_opts['m_lr'] = m_lr
    nn_opts['loss'] = loss
    nn_opts['momentum'] = momentum
    nn_opts['optimizer'] = optimizer
    nn_opts['w_decay'] = w_decay
    return nn_opts


@profile
def reset_model(model, opts):
    loss, lr, _, momentum, optimizer = get_hyper_params(model, opts)
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
        per_class[cls] -= 1
        pairs.append((cls, pred))
        confusion_matrix[cls, label] += 1
        if per_class[cls] == 0:
            remaining_cls.remove(cls)

    return pairs, confusion_matrix


@profile
def clamp_parameters(nn_opts, opts):
    if opts['m']:
        # nn_opts['M'].data.clamp_(min=0)
        pass


@profile
def get_hyper_params(nn_opts, opts):
    loss_func = torch.nn.MultiMarginLoss(margin=1e0)
    w_lr = 0.05 if opts['m'] else 0.01
    m_lr = 0.01
    w_decay = 0.9
    momentum = 0.1
    if opts['m']:
        if nn_opts['parametric_m']:
            m_lr = 0.1
            optim = torch.optim.Adam(nn_opts['M_params'], lr=m_lr)
        else:
            optim = torch.optim.Adam([nn_opts['M']], lr=m_lr)
    else:
        optim = None
    return loss_func, w_lr, m_lr, w_decay, momentum, optim


@profile
def get_nn_options(data, opts):
    no_m_func = 'attr1_func' not in data or data['attr1_func'] == 0
    M = torch.tensor(data['M'], requires_grad=False)
    o, a = len(data['idx'][0]), M.shape[1]
    m_params = []
    if no_m_func:
        if opts['m']:
            u = Uniform(1, 10)
            M = u.sample(M.shape)
            M.requires_grad = True
            M.data.clamp_(min=0)
    else:
        if opts['m']:
            m_params += get_attribute_params(data, opts, 0)
            m_params += get_attribute_params(data, opts, 1)
        else:
            m_params = torch.tensor([data[f'attr1_func_params'][0], data[f'attr2_func_params'][0]], requires_grad=False)

    if opts['w']:
        w = np.ones((a, 1))
        w /= w.sum()
    else:
        w = np.array(data['w'])
    options = {
        'phi1': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['phi1']], requires_grad=False),
        'phi2': torch.tensor([0.01], requires_grad=True) if opts['s']
        else torch.tensor([data['phi2']], requires_grad=False),
        'b': torch.tensor([10.0], requires_grad=True) if opts['s']
        else torch.tensor([float(data['b'])], requires_grad=False),
        'options_count': o,
        'attr_count': a,
        'M': M,
        'M_params': m_params,
        'normalize': data['normalize'] == 1,
        'attr1_func': 0 if no_m_func else data['attr1_func'],
        'attr2_func': 0 if no_m_func else data['attr2_func'],
        'parametric_m': not no_m_func,
        'P0': np.zeros((o, 1)),
        'w': w,
        'sig2': data['sig2'],
        'threshold': data['threshold']
    }

    return options


def get_parametric_attr_values(nn_opts, attr):
    if nn_opts['normalize']:
        if nn_opts[f'attr{attr + 1}_func'] == 1:
            alpha = nn_opts['M_params'][attr]
            return torch.exp(nn_opts['M'][:, attr] * alpha) / torch.exp(nn_opts['M'][:, attr] * alpha).sum()
        if nn_opts[f'attr{attr + 1}_func'] == 2:
            alpha = nn_opts['M_params'][attr]
            return torch.sigmoid(alpha * nn_opts['M'][:, attr]) / torch.sigmoid(alpha * nn_opts['M'][:, attr]).sum()
    else:
        if nn_opts[f'attr{attr + 1}_func'] == 1:
            alpha = nn_opts['M_params'][attr]
            return torch.exp(nn_opts['M'][:, attr] * alpha)
        if nn_opts[f'attr{attr + 1}_func'] == 2:
            alpha = nn_opts['M_params'][attr]
            return torch.sigmoid(alpha * nn_opts['M'][:, attr])
    return None


def get_attribute_params(data, opts, attr):
    if not opts['m']:
        alpha = torch.tensor(data[f'attr{attr + 1}_func_params'], requires_grad=False)
        return [alpha]

    if data[f'attr{attr + 1}_func'] == 1:
        alpha = torch.tensor([0.0])  # Uniform(-5, 5).sample()
        alpha.requires_grad = True
    elif data[f'attr{attr + 1}_func'] == 2:
        alpha = torch.tensor([0.0])  # Uniform(-5, 5).sample()
        alpha.requires_grad = True

    return [alpha]


@profile
def get_model_dist(model, data, n):
    freq_list = []
    for idx in data['idx']:
        M = np.array(model['M'])[idx]
        S = hotaling_S(M, model['phi1'], model['phi2'], model['b'])
        P0 = np.zeros((M.shape[0], 1))
        m = DFT(M, S, np.array(model['w']), P0, np.array(model['sig2']))
        f, converged = get_threshold_based_dft_dist(m, n, model["threshold"], data["relative"])
        freq_list.append(f.squeeze().tolist())
    return freq_list
