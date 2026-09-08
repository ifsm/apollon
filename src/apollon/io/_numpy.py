"""
Numpy binary IO
"""

import pathlib

import numpy as np

from .. typing import Array, PathType


def dump_numpy(data: Array, path: PathType) -> None:
    """Save an array to numpy binary format without using pickle.

    Args:
        data:  Numpy array.
        path:  Path to save the file.
    """
    path = pathlib.Path(path)
    with path.open('wb') as file:
        np.save(file, data, allow_pickle=False)


def load_numpy(path: PathType) -> Array:
    """Load data from numpy's binary format.

    Args:
        path:  File path.

    Returns:
        Data as numpy array.
    """
    path = pathlib.Path(path)
    with path.open('rb') as file:
        data = np.load(file, allow_pickle=False)
    return np.asarray(data)
