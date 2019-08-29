# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/fractal.py

Tools for estimating fractal dimensions.

Function:
    lorenz_attractor   Simulate Lorenz system.
"""
from typing import Tuple

import numpy as np
from scipy import stats
from scipy.spatial import distance

from . types import Array


def log_histogram_bin_edges(dists, n_bins: int, default: float = None):
    """Compute histogram bin edges that are equidistant in log space.
    """
    lower_bound = dists.min()
    upper_bound = dists.max()

    if lower_bound == 0:
        lower_bound = np.absolute(np.diff(dists)).min()

    if lower_bound == 0:
        sd_it = iter(np.sort(dists))
        while not lower_bound:
            lower_bound = next(sd_it)

    if lower_bound == 0:
        lower_bound = np.finfo('float64').eps

    return np.geomspace(lower_bound, dists.max(), n_bins+1)



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
                    metric: str = 'euclidean') -> Array:
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


def embedding_entropy(emb: Array, n_bins: int) -> Array:
    """Compute the information entropy from an embedding.

    Params:
        emb:     Input embedding.
        bins:    Number of bins per dimension.

    Returns:
        Entropy of the embedding.
    """
    counts, edges = np.histogramdd(emb, bins=n_bins)
    return stats.entropy(counts.flatten())


def __lorenz_system(x, y, z, s, r, b):
    """Compute the derivatives of the Lorenz system of coupled
       differential equations.

    Params:
        x, y, z    (float) Current system state.
        s, r, b    (float) System parameters.

    Return:
        xyz_dot    (array) Derivatives of current system state.
    """
    xyz_dot = np.array([s * (y - x),
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
    xyz = np.empty((n, 3))
    xyz[0] = init_xyz

    for i in range(n-1):
        xyz_prime = __lorenz_system(*xyz[i], sigma, rho, beta)
        xyz[i+1] = xyz[i] + xyz_prime * dt

    return xyz
