#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""poisson_core.py
Core functionality for Poisson HMM.
"""

import numpy as _np
from scipy import stats as _stats
from scipy.special import logsumexp as _logsumexp
import warnings
warnings.filterwarnings("ignore")

def log_poisson_fwbw(x, m, _lambda, _gamma, _delta):
    """Compute forward and backward probabilities for Poisson HMM.

    Note: this alogorithm fails if `_delta` has zeros.

    Params:
        x           (np.ndarray)    One-dimensional array of integer values.
        theta       (tuple)         Initial guesses (lambda, gamma, delta).
        maxiter     (int)           Mmaximum number of EM iterations.
        tol         (float)         Convergence criterion.
    """

    n = len(x)
    lalpha, lbeta = _np.zeros((2, n, m))

    # init forward
    pprob = _stats.poisson.pmf(x[:, None], _lambda)
    a_0 = _delta * pprob[0]

    # normalize
    sum_a = a_0.sum()
    a_t = a_0 / sum_a

    # scale factor in log domain
    lscale = _np.log(sum_a)

    # set first forward prob
    lalpha[0] = _np.log(a_t) + lscale

    # start recursion
    for i in range(1, n):
        a_t = a_t @ _gamma * pprob[i]
        sum_a = a_t.sum()
        a_t /= sum_a
        lscale += _np.log(sum_a)
        lalpha[i] = _np.log(a_t) + lscale

    # init backward
    lbeta[-1] = 0
    b_t = _np.repeat(1/m, m)
    lscale = _np.log(m)

    # start backward recursion
    for i in range(n-1, 0, -1):    # ugly reverse iteration in python
        b_t = _gamma @ (pprob[i] * b_t)
        lbeta[i-1] = _np.log(b_t) + lscale
        sum_b = b_t.sum()
        b_t /= sum_b
        lscale += _np.log(sum_b)

    return lalpha, lbeta, _np.log(pprob)


def poisson_EM(x, m, theta, maxiter=1000, tol=1e-6):
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
        lalpha, lbeta, prob = log_poisson_fwbw(x, m, this_lambda, this_gamma, this_delta)

        c = max(lalpha[-1])
        log_likelihood = c + _logsumexp(lalpha[-1] - c)

        for j in range(m):
            for k in range(m):
                next_gamma[j, k] *= _np.sum(_np.exp(lalpha[:n-1, j] +
                                            lbeta[1:n, k] +
                                            prob[1:n, k] -
                                            log_likelihood))
        next_gamma /= _np.sum(next_gamma, axis=1, keepdims=True)

        rab = _np.exp(lalpha + lbeta - log_likelihood)
        next_lambda = (rab * x[:, None]).sum(axis=0) / rab.sum(axis=0)

        next_delta = rab[0] / rab[0].sum()

        crit = (_np.abs(this_lambda - next_lambda).sum() +
                _np.abs(this_gamma - next_gamma).sum()  +
                _np.abs(this_delta - next_delta).sum())

        if crit < tol:
            return next_lambda, next_gamma, next_delta, log_likelihood, True
        else:
            this_lambda = next_lambda.copy()
            this_gamma = next_gamma.copy()
            this_delta = next_delta.copy()

    return next_lambda, next_gamma, next_delta, log_likelihood, False


def poisson_viterbi(mod, x):
    """Calculate the Viterbi path (global decoding) of a PoissonHMM
       given some data x.

       Params:
            x       (array-like) observations
            mod     (HMM-Object)

        Return:
            (np.ndarray) Most probable sequence of hidden states given x.
    """
    n = len(x)

    # Make sure that x is an array
    x = _np.atleast_1d(x)

    # calculate the probability mass for each x_i and for each mean
    pmf_x = _stats.poisson.pmf(x[:, None], mod.lambda_)

    # allocate forward pass array
    xi = _np.zeros((n, mod.m))

    # Probabilities of oberseving x_0 give each state
    probs = mod.delta_ * pmf_x[0]
    xi[0] = probs / probs.sum()

    # Interate over the remaining observations
    for i in range(1, n):
        foo = _np.max(xi[i-1] * mod.gamma_, axis=1) * pmf_x[i]
        xi[i] = foo / foo.sum()

    # allocate backward pass array
    phi = _np.zeros(n, dtype=int)

    # calculate most probable state on last time step
    phi[-1] = _np.argmax(xi[-1])

    # backtrack to first time step
    for i in range(n-2, -1, -1):
        phi[i] = _np.argmax(mod.gamma_[phi[i+1]] * xi[i])

    return phi

