#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""viterbi.py

Generic implementation of the Viterbi algorithm.
"""

import numpy as _np
from scipy import stats as _stats


__author__ = "Michael Blaß"


def viterbi(mod, x):
    '''Calculate the Viterbi path (global decoding) of a HMM
       given some data x.

       Params:
            x       (array-like) observations
            mod     (HMM-Object)

        Return:
            (np.ndarray) Most probable sequence of hidden states given x.
    '''
    n = len(x)

    # Make sure that x is an array
    x = _np.atleast_1d(x)

    # Poisson probabilities of each observation given each poisson mean
    probs = _stats.poisson.pmf(x[:, None], mod.lambda_)

    # Array to hold the path probabilities
    xi = _np.zeros((n, mod.m))

    # Probabilities of oberseving x_0 give each state
    foo = mod.delta_ * probs[0]
    fs = foo.sum()
    fs = fs if fs > 0 else 1.e-20
    xi[0] = foo / fs

    # Interate over the remaining observations
    for i in range(1, n):
        foo = _np.max(xi[i-1] * mod.lambda_, axis=0) * probs[i]
        fs = foo.sum()
        fs = fs if fs > 0 else 1.e-20
        xi[i] = foo / fs

    # Backtracking: get the state number with highest probability
    phi = _np.zeros(n, dtype=int)
    phi[-1] = _np.argmax(xi[-1])
    for i in range(n-2, 0, -1):
        phi[i] = _np.argmax(mod.gamma_[:, phi[i+1]] * xi[i])
    return phi
