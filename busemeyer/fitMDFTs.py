import numpy as np
import math
from distfunct import distfunct
from simMDF import simMDF


def fitMDFTs(x, D, FD, MM, C3, Ns):
    # ccluates predicted and compare to observed

    nc = D.shape[0]
    na = D.shape[1]
    P3 = np.zeros((nc, na))
    TV = np.zeros((nc, 1))

    wgt = math.exp(x[0])
    phi1 = math.exp(x[1])
    phi2 = math.exp(x[2])
    sig2 = math.exp(x[3])
    theta1 = math.exp(x[4])
    w = math.exp(x[5])
    w = np.array([[w], [1 - w]])  # weight vector for m attributes, m=2 in this case

    for i in range(nc):
        M3 = MM[:, :, i]
        G3, EG = distfunct(M3, wgt, phi1, phi2)  # returns gamma
        p3, T = simMDF(G3, C3, M3, w, theta1, sig2, Ns)
        P3[i, :] = p3
        TV[i] = T
    dev = (D - P3)
    sse = (dev * dev).sum()

    return sse, P3, TV
