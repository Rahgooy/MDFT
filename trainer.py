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
    ε = 5e-3
    it = 0
    delta_w = 0
    nn_opts = initi_nn_opts(opts, dataset)
    start = time()
    decay = None
    nc = len(dataset['idx'])  # number of option combinations
    ns = opts['ntrain']  # number of samples
    for it in range(opts['niter']):
        loss = 0
        avg_t = 0
        avg_M = 0
        for j in permutation(nc):
            model = get_nn_model(nn_opts, dataset['idx'][j])
            if decay is None:
                decay = nn_opts['decay']
            else:
                decay = decay ** (it // 10)

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
                avg_M += nn_opts['M'].detach().numpy()

            if opts['w']:
                l.backward()
                W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list]).sum(axis=0) / len(W_list)
                delta_w = nn_opts['momentum'] * delta_w + nn_opts['lr'] * decay * W_grad
                model.w -= delta_w.reshape(-1, 1)
                model.w = model.w.clip(0.1, 0.9)
                model.w /= model.w.sum()

            profiler.finish("calc_grad")

        avg_t /= nc
        error = loss.detach().numpy() / (nc * ns)
        avg_M /= nc
        mdl = {
            "M": nn_opts['M'].data.numpy().copy().tolist(),
            "φ1": float(nn_opts['φ1'].data.numpy().copy()[0]),
            "φ2": float(nn_opts['φ2'].data.numpy().copy()[0]),
            "b": float(nn_opts['b'].data.numpy().copy()[0]),
            "σ2": nn_opts['σ2'],
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

        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or error < ε:
            print_progress(best_model, error, mse, it, opts, time() - start)
            # print([len(predictions[p]) / opts['ntrain'] for p in predictions])
            start = time()

        if mse < ε:
            break

    return best_model, it
