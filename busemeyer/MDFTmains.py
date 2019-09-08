import numpy as np
from time import time
from scipy.optimize import fmin
from fitMDFTs import fitMDFTs
from math import log

start = time()
np.set_printoptions(precision=4, suppress=True)

Ns = 10 ** 4
sw = 1

MM = np.zeros((3, 2, 10))
MM[:, :, 0] = [[1, 3], [3, 1], [.9, 3.1]]  # sim
MM[:, :, 1] = [[1, 3], [3, 1], [1.1, 2.9]]  # sim
MM[:, :, 2] = [[1, 3], [3, 1], [1.75, 2.25]]  # com
MM[:, :, 3] = [[1, 3], [3, 1], [2, 2]]  # comp
MM[:, :, 4] = [[1, 3], [3, 1], [2.25, 1.75]]  # comp
MM[:, :, 5] = [[1, 3], [3, 1], [.5, 2.5]]  # attraction
MM[:, :, 6] = [[1, 3], [3, 1], [1.1, 2.5]]  # attraction
MM[:, :, 7] = [[.5, .5], [.7, .7], [2, 2]]  # dom
MM[:, :, 8] = [[.75, .75], [1, 1], [1.25, 1.25]]  # dom
MM[:, :, 9] = [[.5, .5], [1, 1], [2, 2]]  # dom

D = np.array([
    [.3, .4, .3],
    [.3, .4, .3],
    [.3, .34, .36],
    [.3, .3, .4],
    [.34, .3, .36],
    [.8, .2, 0],
    [.8, .2, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]])

N = 100  # no. observations per choice set
# alternatively enter a new matrix called FD containing the choice frequencies
FD = N * D

na = D.shape[1]  # no of options in choice set

C3 = -1 / (na - 1) * np.ones((na, na))  # C matrix for na options
C3 = C3 - np.diag(np.diag(C3)) + np.eye(na)

#     dis wgt  dist phi1 dist ph2  sig    thresh    w=attention to attribue 1
# x0 = [log(12) log(.022) log(.05) log(1) log(17.5) .5];
# x0 = [6.2197    0.0098    0.0482    1.0007   21.1196 .50]; # SSE = .1176
# x0 = [6.2197    0.00      0.0482    1.0007   21.1196 .50];
# x0 = log(x0);    % real valued matlab parms get exponentiated to positive values
#  x0=[  1.8540   -4.5461   -3.0421    0.0007    3.1017   -0.7082] # .0953
x0 = np.array([1.8776, -4.5708, -3.0938, 0.0007, 3.0900, -0.7021])  # .0821

iter = 1
best_fit = 1000
curr = 0


def fit(x):
    global best_fit, curr
    sse, _, _ = fitMDFTs(x, D, FD, MM, C3, Ns)
    best_fit = min(sse, best_fit)
    curr = sse
    return sse


def progress(x):
    global iter
    print(f"Iteration {iter} - best: {best_fit:.4f}")
    iter += 1


if sw == 1:  # fit data
    x = fmin(lambda X: fit(X), x0, maxiter=100, callback=progress)
    print('Solution')
    print(x)
    print(f'x: {x}')
    print(f'sse: {best_fit:.3f}')
    print(f'wgt = {np.exp(x[0]):.3f}')
    print(f'phi1 = {np.exp(x[1]):.3f}')
    print(f'phi2 = {np.exp(x[2]):.3f}')
    print(f'sig2 = {np.exp(x[3]):.3f}')
    print(f'theta1 = {np.exp(x[4]):.3f}')
    print(f'w0 = {np.exp(x[5]):.3f}')

else:  # bypass fitting
    sse, P3, TV = fitMDFTs(x0, D, FD, MM, C3, Ns)
    print(f'sse: {sse:.3f}')
    print(f'wgt = {np.exp(x0[0]):.3f}')
    print(f'phi1 = {np.exp(x0[1]):.3f}')
    print(f'phi2 = {np.exp(x0[2]):.3f}')
    print(f'sig2 = {np.exp(x0[3]):.3f}')
    print(f'theta1 = {np.exp(x0[4]):.3f}')
    print(f'w = {np.exp(x0[5]):.3f}')

    print(f'sse: {sse:.3f}')
    print('         target data      model predictions       time')
    print(np.hstack((D, P3, TV)))

print(f'Elapsed time is {time() - start: 0.2f} seconds')
