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
    
    
def __lorenz_system(x, y, z, s, r, b):
    """Compute the derivatives of the Lorenz system of coupled
       different equations.
    
    Params:
        x, y, z    (float) Current system state.
        s, r, b    (float) System parameters.
    
    Return:
        xyz_dot    (array) Derivatives of current system state.
    """
    xyz_dot = _np.array([s * (y - x),
                          x * (r - z) - y,
                          x * y - b * z])
    return xyz_dot


def lorenz_attractor(n, sigma=10, rho=28, beta=8/3,
                     init_xyz=(0., 1., 1.05), dt=0.01):
    """Simulate a Lorenz system.
    
    Params:
        n        (int)   Number of data points to generate.
        sigma    (float) System parameter.
        rho      (rho)   System parameter.
        beta     (beta)  System parameter.
        init_xyz (tuple) Initial System state.
        dt       (float) Step size.
                     
    Return:
        xyz    (array) System states.
    """
    xyz = _np.empty((n, 3))
    xyz[0] = init_xyz

    for i in range(n-1):
        xyz_prime = __lorenz_system(*xyz[i], sigma, rho, beta)
        xyz[i+1] = xyz[i] + xyz_prime * dt
        
    return xyz
