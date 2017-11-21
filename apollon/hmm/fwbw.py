#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""fwbw.py
Implementation of the forward backward algorithm."""


import numpy as _np
from scipy import stats as _stats


def forward_backward(x, m, _lambda, _gamma, _delta):
    """Calculate the forward and backward probabilities."""

    n = len(x)
    alpha, beta = _np.zeros((2, n, m))

    # init forward
    allprobs = _stats.poisson.pmf(*_np.ix_(x, _lambda))
    a_0 = _delta * allprobs[0]

    # normalize
    sum_a = a_0.sum()
    a_t = a_0 / sum_a

    # scale factor in log domain
    lscale = _np.log(sum_a)

    # set first forward prob
    alpha[0] = _np.log(a_t) + lscale

    # start recursion
    for i in range(1, n):
        a_t = a_t @ _gamma * allprobs[i]
        sum_a = a_t.sum()
        a_t /= sum_a
        lscale += _np.log(sum_a)
        alpha[i] = _np.log(a_t) + lscale

    # init backward
    beta[-1] = 0
    b_t = _np.repeat(1 / m, m)
    lscale = _np.log(m)

    # start backward recursion
    for i in range(n-2, -1, -1):    # ugly reverse iteration in python
        b_t = _gamma @ (allprobs[i+1] * b_t)
        beta[i] = _np.log(b_t) + lscale
        sum_b = b_t.sum()
        b_t /= sum_b
        lscale += _np.log(sum_b)
    return alpha, beta
