"""
General I/O functionallity
"""

import pathlib
import pickle
from typing import Any

import numpy as np

from .. typing import Array, PathType
from . json import ArrayEncoder

def generate_outpath(in_path: PathType,
                     out_path: PathType | None,
                     suffix: str | None = None) -> PathType:
    """Generate file paths for feature output

    If ``out_path`` is ``None``, the basename of ``in_path`` is taken
    with the extension replaced by ``suffix``.

    Args:
        in_path:   Path to file under analysis
        out_path:  Commandline argument
        suffix:    File extension

    Returns:
        Valid output path
    """
    in_path = pathlib.Path(in_path)
    if suffix is None:
        default_fname = f'{in_path.stem}'
    else:
        default_fname = f'{in_path.stem}.{suffix}'

    if out_path is None:
        out_path = pathlib.Path(default_fname)
    else:
        out_path = pathlib.Path(out_path)
        if not out_path.suffix:
            out_path = out_path.joinpath(default_fname)
        if not out_path.parent.is_dir():
            msg = f'Error. Path "{out_path.parent!s}" does not exist.'
            raise ValueError(msg)
    return out_path



def load_from_pickle(path: PathType) -> Any:
    """Load a pickled file

    Args:
        path:  Path to file.

    Returns:
        Unpickled object
    """
    path = pathlib.Path(path)
    with path.open('rb') as file:
        data = pickle.load(file)
    return data


def repath(current_path: PathType, new_path: PathType,
           ext: str | None = None) -> PathType:
    """Change the path and keep the file name

    Optinally change the extension, too.

    Args:
        current_path:  The path to change.
        new_path:      The new path.
        ext:           Change file extension if ``ext`` is not None.

    Returns:
        New path.
    """
    current_path = pathlib.Path(current_path)
    new_path = pathlib.Path(new_path)
    if ext is None:
        new_path = new_path.joinpath(current_path.name)
    else:
        ext = ext if ext.startswith('.') else '.' + ext
        new_path = new_path.joinpath(current_path.stem + ext)
    return new_path


def save_to_pickle(data: Any, path: PathType) -> None:
    """Pickles data to path.

    Args:
        data:  Pickleable object.
        path:  Path to save the file.
    """
    path = pathlib.Path(path)
    with path.open('wb') as file:
        pickle.dump(data, file)


def save_to_npy(data: Array, path: PathType) -> None:
    """Save an array to numpy binary format without using pickle.

    Args:
        data:  Numpy array.
        path:  Path to save the file.
    """
    path = pathlib.Path(path)
    with path.open('wb') as file:
        np.save(file, data, allow_pickle=False)


def load_from_npy(path: PathType) -> Array:
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
