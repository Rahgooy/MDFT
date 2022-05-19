from time import time

import numpy as np
from numpy.random.mtrand import permutation

from mdft_nn.helpers.profiling import profile, global_profiler as profiler
from mdft_nn.trainer_helpers import *
import torch


@profile
def train(dataset, opts):
    best_model = None
    best_mse = float('inf')
    eps = 1e-4
    it = 0
    print(f"w = {dataset['w']}")
    nn_opts = initi_nn_opts(opts, dataset)
    start = time()
    nc = len(dataset['idx'])  # number of option combinations
    ns = opts['ntrain']  # number of samples
    for it in range(opts['niter']):
        loss = 0
        avg_t = 0
        for j in permutation(nc):
            pairs, a_t = forward_pass(
                nn_opts, dataset, j, it, opts, ns)
            if len(pairs) == 0:
                continue
            l = compute_loss(pairs, nn_opts)
            loss += l
            avg_t += a_t
            backward_pass(opts, nn_opts, l)

        avg_t /= nc
        error = (loss.detach().numpy() if type(loss) == torch.Tensor else loss) / (nc * ns)
        mdl = {
            "M": nn_opts['M'].data.numpy().copy().tolist(),
            "phi1": float(nn_opts['phi1'].data.numpy().copy()[0]),
            "phi2": float(nn_opts['phi2'].data.numpy().copy()[0]),
            "b": float(nn_opts['b'].data.numpy().copy()[0]),
            "sig2": nn_opts['sig2'],
            "threshold": nn_opts['threshold'],
            "w": torch.softmax(nn_opts['w'], dim=0).detach().numpy().copy().tolist(),
            "iter": it + 1,
            "avg_t": avg_t
        }
        dist = get_model_dist(mdl, dataset, 1000)
        mse = 0
        for i in range(nc):
            m = np.array(dataset['D'][i]) - np.array(dist[i])
            mse += (m*m).sum() / nc
        mdl['mse'] = mse
        if mse < best_mse:
            best_mse = mse
            best_model = mdl

        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or error < eps:
            print_progress(best_model, error, mse, it, opts, time() - start)
            start = time()

        if mse < eps:
            break

    return best_model, it


def backward_pass(opts, nn_opts, l):
    profiler.start("calc_grad")
    nn_opts['optimizer'].zero_grad()
    l.backward()
    if opts['m']:
        torch.nn.utils.clip_grad_norm_(nn_opts['M'], nn_opts['grad_clip'])
    nn_opts['optimizer'].step()
    clamp_parameters(nn_opts, opts)

    profiler.finish("calc_grad")


def forward_pass(nn_opts, dataset, j, it, opts, ns):
    model = get_nn_model(nn_opts, dataset['idx'][j])
    per_class_samples = get_per_class_samples(model, dataset['D'][j], ns)
    predictions, a_t, _ = get_model_predictions(
        model, opts['w'], ns, dataset['pref_based'])
    pairs, _ = align_samples(per_class_samples, predictions)
    return pairs, a_t
