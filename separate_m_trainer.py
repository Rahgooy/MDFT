from time import time

import numpy as np
from numpy.random.mtrand import permutation

from helpers.profiling import profile, global_profiler as profiler
from trainer_helpers import initialize, update_fixed_parameters, get_per_class_samples, get_model_predictions, align_samples, \
    compute_loss, print_progress, clamp_parameters


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
        loss = 0
        avg_t = 0
        for j in permutation(len(dataset['M'])):
            if model is None:  # Initialize
                model = initialize(dataset, opts, j)
                # print("Learning rate : {}".format(opts['lr']))
            else:
                update_fixed_parameters(dataset, model, opts, j)

            per_class_samples = get_per_class_samples(dataset, j, model, opts)
            predictions, W_list, a_t, max_t = get_model_predictions(model, opts)
            avg_t += a_t
            pairs, confusion_matrix = align_samples(per_class_samples, predictions)
            l = compute_loss(opts, pairs, model)
            loss += l
            if opts['w']:
                l.backward()
                W_grad = np.array([x.grad.detach().numpy().sum(axis=1) for x in W_list]).sum(axis=0) / len(W_list)
                delta_w = opts['momentum'] * delta_w + opts['lr'] * W_grad
                model.w -= delta_w.reshape(-1, 1)
                model.w = model.w.clip(0.1, 0.9)
                model.w /= model.w.sum()

        avg_t /= len(dataset['M'])
        error = loss.detach().numpy() / (opts['ntrain'] * len(dataset['M']))
        if error <= best_error:
            best_error = error
            best_model = {
                "M": model.M.data.numpy().copy().tolist(),
                "φ1": float(model.φ1.data.numpy().copy()[0]),
                "φ2": float(model.φ2.data.numpy().copy()[0]),
                "b": float(model.b.data.numpy().copy()[0]),
                "σ2": model.σ2,
                "threshold": model.threshold,
                "w": model.w.copy().tolist(),
                "error": best_error,
                "iter": it + 1,
                "avg_t": avg_t
            }
            no_progress_it = 0

        if it % opts['nprint'] == 0 or it == opts['niter'] - 1 or error < ε:
            print_progress(best_model, error, it, model, opts, time() - start)
            # print([len(predictions[p]) / opts['ntrain'] for p in predictions])
            start = time()

        if error < ε:
            break

        profiler.start("calc_grad")

        if opts['m']:
            opts['optimizer'].zero_grad()
            loss.backward()
            opts['optimizer'].step()
            clamp_parameters(model, opts)
            model.update_S()

        profiler.finish("calc_grad")

        no_progress_it += 1

        if no_progress_it > 100:
            model = None
            no_progress_it = 0

    return best_model, it