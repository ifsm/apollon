#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""em.py

Expectation Maximization algorithm for HMMs with
Poisson distributed tate-depended variables.
"""

import numpy as _np
from scipy import stats as _stats
from scipy.special import logsumexp as _logsumexp

from fwbw import forward_backward


def EM(x, m, theta, maxiter=1000, tol=1e-6):
    """Estimate the parameters of an m-state PoissonHMM.

    Params:
        x           (np.ndarray)    One-dimensional array of integer values.
        theta       (tuple)         Initial guesses (lambda, gamma, delta).
        maxiter     (int)           Mmaximum number of EM iterations.
        tol         (float)         Convergence criterion.
    """
    n = len(x)

    this_lambda = theta[0].copy()
    this_gamma = theta[1].copy()
    this_delta = theta[2].copy()

    next_lambda = theta[0].copy()
    next_gamma = theta[1].copy()
    next_delta = theta[2].copy()

    for i in range(maxiter):
        alpha, beta, prob = forward_backward(x, m, this_lambda, this_gamma, this_delta)

        c = max(alpha[-1])
        log_likelihood = c + _logsumexp(alpha[-1] - c)

        for j in range(m):
            for k in range(m):
                next_gamma[j, k] *= _np.sum(_np.exp(alpha[:n-1, j] +
                                            beta[1:n, k] +
                                            prob[1:n, k] -
                                            log_likelihood))
        next_gamma /= _np.sum(next_gamma, axis=1)

        rab = _np.exp(alpha + beta - log_likelihood)
        next_lambda = (rab * x[:, None]).sum(axis=0) / rab.sum(axis=0)

        next_delta = rab[0] / rab[0].sum()

        crit = (_np.abs(this_lambda - next_lambda).sum() +
                _np.abs(this_gamma - next_gamma).sum()  +
                _np.abs(this_delta - next_delta).sum())

        if crit < tol:
            return this_lambda, this_gamma, this_delta, i, log_likelihood

        this_lambda = next_lambda.copy()
        this_gamma = next_gamma.copy()
        this_delta = next_delta.copy()

    return False

