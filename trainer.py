from time import time

import numpy as np
from numpy.random.mtrand import permutation

from helpers.profiling import profile, global_profiler as profiler
from trainer_helpers import initi_nn_opts, get_nn_model, get_per_class_samples, get_model_predictions, \
    align_samples, \
    compute_loss, print_progress, clamp_parameters, get_model_dist


@profile
def train(dataset, opts):
    best_model = None
    best_mse = 1e100
    eps = 1e-4
    it = 0
    delta_w = 0
    print(f"w = {dataset['w']}")
    nn_opts = initi_nn_opts(opts, dataset)
    start = time()
    w_decay = None
    nc = len(dataset['idx'])  # number of option combinations
    ns = opts['ntrain']  # number of samples
    for it in range(opts['niter']):
        loss = 0
        avg_t = 0
        for j in permutation(nc):
            model = get_nn_model(nn_opts, dataset['idx'][j])
            if w_decay is None:
                w_decay = nn_opts['w_decay']
            else:
                w_decay = nn_opts['w_decay'] ** (it // (opts['niter'] / 10))

            per_class_samples = get_per_class_samples(model, dataset['D'][j], ns)
            predictions, W_list, a_t, max_t = get_model_predictions(model, opts['w'], ns)
            avg_t += a_t
            pairs, confusion_matrix = align_samples(per_class_samples, predictions)
            l = compute_loss(pairs, nn_opts)
            loss += l

            profiler.start("calc_grad")
            if opts['m']:
                nn_opts['optimizer'].zero_grad()
                l.backward()
                nn_opts['optimizer'].step()
                clamp_parameters(nn_opts, opts)

            if opts['w']:
                if not opts['m']:
                    l.backward()
                W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list])\
                             .sum(axis=0).reshape(-1, 1) / len(W_list)
                delta_w = nn_opts['momentum'] * delta_w + nn_opts['w_lr'] * w_decay * W_grad
                nn_opts['w'] -= delta_w
                nn_opts['w'] = nn_opts['w'].clip(0.2, 0.8)
                nn_opts['w'] /= nn_opts['w'].sum()

            profiler.finish("calc_grad")

        avg_t /= nc
        error = loss.detach().numpy() / (nc * ns)
        mdl = {
            "M": nn_opts['M'].data.numpy().copy().tolist(),
            "phi1": float(nn_opts['phi1'].data.numpy().copy()[0]),
            "phi2": float(nn_opts['phi2'].data.numpy().copy()[0]),
            "b": float(nn_opts['b'].data.numpy().copy()[0]),
            "sig2": nn_opts['sig2'],
            "threshold": nn_opts['threshold'],
            "w": nn_opts['w'].copy().tolist(),
            "iter": it + 1,
            "avg_t": avg_t
        }
        dist = get_model_dist(mdl, dataset, 1000)
        mse = np.array(dataset['D']) - np.array(dist)
        mse = (mse * mse).sum() / nc
        mdl['mse'] = mse
        if mse < best_mse:
            best_mse = mse
            best_model = mdl

        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or error < eps:
            print_progress(best_model, error, mse, it, opts, time() - start)
            # print([len(predictions[p]) / opts['ntrain'] for p in predictions])
            start = time()

        if mse < eps:
            break

    return best_model, it
