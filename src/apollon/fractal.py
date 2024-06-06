"""
Tools for estimating fractal dimensions
"""

import numpy as np
from scipy import stats
from scipy.spatial import distance

from . types import Array, FloatArray, floatarray


def delay_embedding(inp: Array, delay: int, m_dim: int) -> FloatArray:
    """Compute a delay embedding of the `inp`

    This method makes a hard cut at the upper bound of `inp` and
    does not perform zero padding to match the input size.

    Args:
        inp:   One-dimensional input vector
        delay: Vector delay in samples
        m_dim: Number of embedding dimension

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
                    metric: str = 'euclidean') -> FloatArray:
    """Perfom a delay embedding and return the pairwaise distances
    of the delayed vectors

    The returned vector is the flattend upper triangle of the distance
    matrix.

    Args:
        inp:    One-dimensional input vector
        delay:  Vector delay in samples
        m_dim   Number of embedding dimension
        metric: Metric to use

    Returns:
        Flattened upper triangle of the distance matrix
    """
    emb_vects = delay_embedding(inp, delay, m_dim)
    return floatarray(distance.pdist(emb_vects, metric))


def embedding_entropy(emb: Array, n_bins: int) -> FloatArray:
    """Compute the information entropy from an embedding

    Args:
        emb:    Input embedding
        bins:   Number of bins per dimension

    Returns:
        Entropy of the embedding
    """
    counts, _ = np.histogramdd(emb, bins=n_bins)
    return floatarray(stats.entropy(counts.flatten()))


def lorenz_system(state: tuple[float, float, float],
                  params: tuple[float, float, float]
                  ) -> FloatArray:
    """Compute the derivatives of the Lorenz system of coupled
       differential equations

    Args:
        state:  Current system state
        params: Szstem parameters sigma, rho, beta

    Return:
        xyz_dot    Derivatives of current system state
    """
    # pylint: disable=invalid-name
    x, y, z = state
    sigma, rho, beta = params
    xyz_dot = np.array([sigma * (y - x),
                        x * (rho - z) - y,
                        x * y - beta * z], dtype=np.double)
    return xyz_dot


def lorenz_attractor(n_samples: int, params: tuple[float, float, float] = (10.0, 28.0, 8/3),
                     init_state: tuple[float, float, float] = (0., 1., 1.05),
                     diff_t: float = 0.01) -> FloatArray:
    """Simulate the Lorenz system

    Args:
        n_samples:   Number of data points to generate
        params:      System parameters sigma, rho, beta
        init_state:  Initial System state
        dt:          Positive step size

    Return:
        xyz: System state
    """
    if n_samples < 0:
        raise ValueError("``n_samples`` must be positive integer")

    if diff_t < 0:
        raise ValueError("``diff_t`` must be positive float")

    xyz = np.empty((n_samples, 3), dtype=np.double)
    xyz[0] = init_state

    for i in range(n_samples-1):
        xyz_prime = lorenz_system(xyz[i], params)
        xyz[i+1] = xyz[i] + xyz_prime * diff_t

    return xyz
