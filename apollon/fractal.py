#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""apollon/fractal.py   (c) Michael Blaß 2017

Tools for estimating fractal dimensions.

Function:
    embdedding      Pseudo-phase space embdedding.
    pps_entropy     Entropy of pps embdedding.
"""

import numpy as _np
from scipy import stats as _stats


def embedding(inp_sig, tau, m=2, mode='zero'):
    """Generate n-dimensional pseudo-phase space embedding.

    Params:
        inp_sig    (iterable) Input signal.
        tau        (int) Time shift.
        m          (int) Embedding dimensions.
        mode       (str) Either `zero` for zero padding,
                                `wrap` for wrapping the signal around, or
                                `cut`, which cuts the signal at the edges.
                         Note: In cut-mode, each dimension is only
                               len(sig) - tau * (m - 1) samples long.
    Return:
        (np.ndarray) of shape
                        (m, len(inp_sig)) in modes 'wrap' or 'zeros', or
                        (m, len(sig) - tau * (m - 1)) in cut-mode.
    """
    inp_sig = _np.atleast_1d(inp_sig)
    N = len(inp_sig)

    if mode == 'zero':
        # perform zero padding
        out = _np.zeros((m, N))
        out[0] = inp_sig
        for i in range(1, m):
            out[i, tau*i:] = inp_sig[:-tau*i]

    elif mode == 'wrap':
        # wraps the signal around at the bounds
        out = _np.empty((m, N))
        for i in range(m):
            out[i] = _np.roll(inp_sig, i*tau)

    elif mode == 'cut':
        # cut every index beyond the bounds
        Nm = N - tau * (m-1)    # number of vectors
        if Nm < 1:
            raise ValueError('Embedding params to large for input.')
        out = _np.empty((m, Nm))
        for i in range(m):
            off = N - i * tau
            out[i] = inp_sig[off-Nm:off]

    else:
        raise ValueError('Unknown mode `{}`.'.format(pad))

    return out


def pps_entropy(emb, bins):
    """Calculate entropy of given embedding unsing log_e.

    Params:
        emb    (ndarray) pps embedding.
        bins   (int) Number of histogram bins per axis.""

    Return:
        (float) Entropy of pps.
    """    
    pps, _ = _np.histogramdd(emb.T, bins=bins)
    H = _stats.entropy(pps.flat) / _np.log(pps.size)
    return H
