# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""apollon/io.py -- General I/O functionallity.

Classes:
    ArrayEncoder            Serialize numpy array to JSON.
    FileAccessControl       Descriptor for file name attributes.

Functions:
    array_print_opt         Set format for printing numpy arrays.
    decode_array            Decode numpy array from JSON.
    files_in_folder         Iterate over all files in given folder.
    load                    Load pickled data.
    repath                  Change path but keep file name.
    save                    Pickle some data.
"""
from contextlib import contextmanager as _contextmanager
import json as _json
import pathlib as _pathlib
import pickle
import typing

import numpy as _np

from . import types as _types


class ArrayEncoder(_json.JSONEncoder):
    # pylint: disable=E0202
    # Issue: False positive for E0202 (method-hidden) #414
    # https://github.com/PyCQA/pylint/issues/414
    """Encode np.ndarrays to JSON.

    Simply set the `cls` parameter of the dump method to this class.
    """
    def default(self, o):
        """Custon default JSON encoder. Properly handles numpy arrays and JSONEncoder.default
        for all other types.

        Params:
            o (any)  Object to encode.

        Returns:
            (dict)
        """
        if isinstance(o, _np.ndarray):
            out = {'__ndarray__': True,
                   '__dtype__': o.dtype.str,
                   'data': o.astype('float64').tolist()}
            return out
        return _json.JSONEncoder.default(self, o)


def decode_array(json_data: dict) -> typing.Any:
    """Properly decodes numpy arrays from a JSON data stream.

    This method need to be called on the return value of ``json.load`` or ``json.loads``.

    Args:
        json_data (dict)    JSON formatted dict to encode.

    Returns:
        (any)
    """
    if '__ndarray__' in json_data and '__dtype__' in json_data:
        return _np.array(json_data['data'], dtype=json_data['__dtype__'])
    return json_data


def generate_outpath(in_path: str, out_path: str = None,
                     suffix: str = None) -> str:
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
    in_path = _pathlib.Path(in_path)
    if suffix is None:
        default_fname = '{}'.format(in_path.stem)
    else:
        default_fname = '{}.{}'.format(in_path.stem, suffix)

    if out_path is None:
        out_path = _pathlib.Path(default_fname)
    else:
        out_path = _pathlib.Path(out_path)
        if not out_path.suffix:
            out_path = out_path.joinpath(default_fname)
        if not out_path.parent.is_dir():
            print('Error. Path "{!s}" does not exist.'.format(out_path.parent))
            exit(10)
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


def dump_json(obj, path: _types.PathType = None) -> None:
    """Write ``obj`` to JSON.

    This function can handel numpy arrays.

    If ``path`` is None, this fucntion writes to stdout.  Otherwise, encoded
    object is written to ``path``.

    Args:
        obj  (any)         Object to be encoded.
        path (PathType)    Output file path.
    """
    if path is None:
        print(_json.dumps(obj, cls=ArrayEncoder))
    else:
        path = _pathlib.Path(path)
        with path.open('w') as json_file:
            _json.dump(obj, json_file, cls=ArrayEncoder)


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
            _path = _pathlib.Path(file_name).resolve()
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
    https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-numpy-array-without-
    scientific-notation-and-with-given-pre
    """
    std_options = _np.get_printoptions()
    _np.set_printoptions(*args, **kwargs)

    try:
        yield
    finally:
        _np.set_printoptions(**std_options)


def load(path: _types.PathType) -> typing.Any:
    """Load a pickled file.

    Args:
        path    (str) Path to file.

    Returns:
        (object) unpickled object
    """
    path = _pathlib.Path(path)
    with path.open('rb') as file:
        data = pickle.load(file)
    return data

def repath(current_path: _types.PathType, new_path: _types.PathType,
           ext: str = None) -> _types.PathType:
    """Change the path and keep the file name. Optinally change the extension, too.

    Args:
        current_path (str or Path)  The path to change.
        new_path     (str or Path)  The new path.
        ext          (str or None)  Change file extension if ``ext`` is not None.

    Returns:
        (pathlib.Path)
    """
    current_path = _pathlib.Path(current_path)
    new_path = _pathlib.Path(new_path)
    file_path = current_path.stem

    if ext is not None:
        if not ext.startswith('.'):
            ext = '.' + ext
        file_path.join(ext)

    return new_path.joinpath(file_path)


def save(data: typing.Any, path: _types.PathType):
    """Pickles data to path.

    Args:
        data    (Any)         Pickleable object.
        path    (str or Path) Path to safe the file.
    """
    path = _pathlib.Path(path)
    with path.open('wb') as file:
        pickle.dump(data, file)
