"""apollon/io.py -- General I/O functionallity.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net

Classes:
    FileAccessControl       Descriptor for file name attributes.

Functions:
    array_print_opt         Set format for printing numpy arrays.
    files_in_folder         Iterate over all files in given folder.
    generate_outpath        Compute path for feature output.
    load_from_pickle        Load pickled data.
    repath                  Change path but keep file name.
    save_to_pickle          Pickle some data.
"""
from contextlib import contextmanager as _contextmanager
import pathlib
import pickle
from typing import Any, Optional

import numpy as np

from .. types import Array, PathType
from . json import ArrayEncoder

def generate_outpath(in_path: PathType,
                     out_path: Optional[PathType],
                     suffix: str = None) -> PathType:
    """Generates file paths for feature und HMM output files.

    If ``out_path`` is ``None``, the basename of ``in_path`` is taken
    with the extension replaced by ``suffix``.

    Args:
        in_path:   Path to file under analysis.
        out_path:  Commandline argument.
        suffix:    File extension.

    Returns:
        Valid output path.
    """
    in_path = pathlib.Path(in_path)
    if suffix is None:
        default_fname = '{}'.format(in_path.stem)
    else:
        default_fname = '{}.{}'.format(in_path.stem, suffix)

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

class PoissonHmmEncoder(ArrayEncoder):
    """JSON encoder for PoissonHmm.
    """
    def default(self, o):
        """Custon default JSON encoder. Properly handles <class 'PoissonHMM'>.

        Note: Falls back to ``ArrayEncoder`` for all types that do not implement
        a ``to_dict()`` method.

        Params:
            o (any)  Object to encode.

        Returns:
            (dict)
        """
        if isinstance(o, HMM):
            items = {}
            for attr in o.__slots__:
                try:
                    items[attr] = getattr(o, attr).to_dict()
                except AttributeError:
                    items[attr] = getattr(o, attr)
            return items
        return ArrayEncoder.default(self, o)

class WavFileAccessControl:
    """Control initialization and access to the ``file`` attribute of class:``AudioData``.

    This assures that the path indeed points to a file, which has to be a .wav file. Otherwise
    an error is raised. The path to the file is saved as absolute path and the attribute is
    read-only.
    """

    def __init__(self):
        """Hi there!"""
        self.__attribute = {}

    def __get__(self, obj, objtype):
        return self.__attribute[obj]

    def __set__(self, obj, file_name):
        if obj not in self.__attribute.keys():
            _path = pathlib.Path(file_name).resolve()
            if _path.exists():
                if _path.is_file():
                    if _path.suffix == '.wav':
                        self.__attribute[obj] = _path
                    else:
                        raise IOError('`{}` is not a .wav file.'
                                      .format(file_name))
                else:
                    raise IOError('`{}` is not a file.'.format(file_name))
            else:
                raise FileNotFoundError('`{}` does not exists.'
                                        .format(file_name))
        else:
            raise AttributeError('File name cannot be changed.')

    def __delete__(self, obj):
        del self.__attribute[obj]


@_contextmanager
def array_print_opt(*args, **kwargs):
    """Set print format for numpy arrays.

    Thanks to unutbu:
    https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-
    numpy-array-without-scientific-notation-and-with-given-pre
    """
    std_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**std_options)


def load_from_pickle(path: PathType) -> Any:
    """Load a pickled file.

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
           ext: Optional[str] = None) -> PathType:
    """Change the path and keep the file name. Optinally change the extension, too.

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
    return data

