"""
Tools for estimating fractal dimensions
"""

import numpy as np
from scipy import stats
from scipy.spatial import distance

from . typing import Array, FloatArray, floatarray


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
