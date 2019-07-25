# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""apollon/fractal.py

Tools for estimating fractal dimensions.

Function:
    lorenz_attractor   Simulate Lorenz system.
"""
from typing import Tuple

import numpy as np
from scipy import stats as _stats
from scipy.spatial import distance

from . types import Array


def delay_embedding(inp: Array, delay: int, m_dim: int) -> Array:
    """Compute a delay embedding of the `inp`.

    This method makes a hard cut at the upper bound of `inp` and
    does not perform zero padding to match the input size.

    Params:
        inp:   One-dimensional input vector.
        delay: Vector delay in samples.
        m_dim: Number of embedding dimension.

    Returns:
        Two-dimensional delay embedding array in which the nth row
        represents the  n * `delay` samples delayed vector.
    """
    max_idx = inp.size - ((m_dim-1)*delay)
    emb_vects = np.empty((max_idx, m_dim))
    for i in range(max_idx):
        emb_vects[i] = inp[i:i+m_dim*delay:delay]
    return emb_vects


def embedding_dists(inp: Array, delay: int, m_dim: int,
                    metric: str = 'sqeuclidean') -> Array:
    """Perfom a delay embedding and return the pairwaise distances
    of the delayed vectors.

    The returned vector is the flattend upper triangle of the distance
    matrix.

    Params:
        inp:    One-dimensional input vector.
        delay:  Vector delay in samples.
        m_dim   Number of embedding dimension.
        metric: Metric to use.

    Returns:
        Flattened upper triangle of the distance matrix.
    """
    emb_vects = delay_embedding(inp, delay, m_dim)
    return distance.pdist(emb_vects, metric)


def correlation_hist(data: Array, delay: int, m_dim: int, n_bins: int,
                     metric: str = 'sqeuclidean') -> Tuple[Array, Array]:
    """Compute histogram of distances in a delay embedding.

    Bin sizes increase logarithmically between the minimal and maximal
    distance in the embedding.

    Params:
        data:    One-dimensional input vector.
        delay:  Vector delay in samples.
        m_dim   Number of embedding dimension.
        n_bins: Number of histogram bins.
        metric: Metric to use.

    Returns:
        Uupper bin edges and number of points per bin.
    """
    dists = embedding_dists(data, delay, m_dim, metric)
    rr = np.geomspace(dists.min(), dists.max(), n_bins)
    cs, rr = np.histogram(dists, rr, density=True)
    return rr[1:], cs


def log_correlation_sum(rr: Array, cs: Array) -> Tuple[Array, Array]:
    "Transform"
    return np.log(rr), np.log(cs.cumsum() / cs.sum())


def correlation_dimension(data: Array, delay: int, m_dim: int, n_bins: int,
        metric: str = 'sqeuclidean', debug: bool = False) -> float:
    """Compute an estimate of the fractal correlation dimension of `data`.

    Params:
        inp:    One-dimensional input vector.
        delay:  Vector delay in samples.
        m_dim   Number of embedding dimension.
        metric: Metric to use.
        debug:  If True, plot visualisation of the estimation process.

    Returns:
        Estimate of the correlation dimension.
    """
    rr, cs = correlation_hist(data, delay, m_dim, n_bins, metric)
    lr, lc = log_correlation_sum(rr, cs)

    lsb = n_bins//3
    usb = n_bins*2//3

    search = slice(n_bins//3, n_bins*2//3)

    scaling_start = lsb + cs[search].argmax()
    scaling_stop = scaling_start + 10

    scaling = slice(scaling_start, scaling_stop)

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(lr, lc)

        vlines(lr[lsb], lc.min(), lc.max(), colors='r')
        vlines(lr[usb], lc.min(), lc.max(), colors='r')

        vlines(lr[scaling_start], lc.min(), lc.max())
        vlines(lr[scaling_stop], lc.min(), lc.max())
        plt.show()

    cdim, err = np.polyfit(lr[scaling], lc[scaling], 1)
    return cdim


def __lorenz_system(x, y, z, s, r, b):
    """Compute the derivatives of the Lorenz system of coupled
       differential equations.

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
    """Simulate a Lorenz system with given parameters.

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
