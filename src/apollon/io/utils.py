"""
Misc IO utilities
"""
import pathlib
from typing import Any, Generator
from contextlib import contextmanager

import numpy as np

from .. typing import PathType

@contextmanager
def array_print_opt(*args: Any, **kwargs: Any) -> Generator[None, None, None]:
    """Set print format for numpy arrays

    Thanks to unutbu:
    https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-
    numpy-array-without-scientific-notation-and-with-given-pre

    Args:
        *args:     Forwarded to ``numpy.set_printoptions``.
        **kwargs:  Forwarded to ``numpy.set_printoptions``.
    """
    std_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**std_options)


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
