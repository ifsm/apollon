"""
Tools for estimating fractal dimensions
"""

import numpy as np
from scipy import stats
from scipy.spatial import distance

from . types import FloatArray, NDArray


def delay_embedding(inp: NDArray, delay: int, m_dim: int) -> NDArray:
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


def embedding_dists(inp: NDArray, delay: int, m_dim: int,
                    metric: str = 'euclidean') -> NDArray:
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
    return distance.pdist(emb_vects, metric)


def embedding_entropy(emb: NDArray, n_bins: int) -> NDArray:
    """Compute the information entropy from an embedding

    Args:
        emb:    Input embedding
        bins:   Number of bins per dimension

    Returns:
        Entropy of the embedding
    """
    counts, edges = np.histogramdd(emb, bins=n_bins)
    return stats.entropy(counts.flatten())


def lorenz_system(state: tuple[float, float, float],
                  sigma: float, rho: float, beta: float) -> FloatArray:
    """Compute the derivatives of the Lorenz system of coupled
       differential equations

    Args:
        state:  Current system state
        sigma:  System parameter
        rho:    System parameter
        beta:   System parameter

    Return:
        xyz_dot    Derivatives of current system state
    """
    x, y, z = state
    xyz_dot = np.array([sigma * (y - x),
                        x * (rho - z) - y,
                        x * y - beta * z], dtype=np.float64)
    return xyz_dot


def lorenz_attractor(n_samples: int, sigma: float = 10.0, rho: float = 28.0,
                     beta: float = 8/3,
                     init_state: tuple[float, float, float] = (0., 1., 1.05),
                     dt: float = 0.01) -> FloatArray:
    """Simulate the Lorenz system

    Args:
        n_samples:   Number of data points to generate
        sigma:       System parameter
        rho:         System parameter
        beta:        System parameter
        init_state:  Initial System state
        dt:          Positive step size

    Return:
        xyz: System state
    """
    if n_samples < 0:
        raise ValueError("``n_samples`` must be positive integer")

    if dt < 0:
        raise ValueError("``dt`` must be positive float")

    xyz = np.empty((n_samples, 3), dtype=np.float64)
    xyz[0] = init_state

    for i in range(n_samples-1):
        xyz_prime = lorenz_system(xyz[i], sigma, rho, beta)
        xyz[i+1] = xyz[i] + xyz_prime * dt

    return xyz
