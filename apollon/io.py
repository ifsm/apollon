"""apollon/IO.py

Tools for file access.

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
import json
import pathlib
import pickle
import typing

import numpy as _np

import apollon.types as apt


class ArrayEncoder(json.JSONEncoder):
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
        if isinstance(o, apt.Array):
            out = {'__ndarray__': True,
                   '__dtype__': o.dtype.str,
                   'data': o.tolist()}
            return out
        return json.JSONEncoder.default(self, o)


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
        return io.ArrayEncoder.default(self, o)


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
    https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-numpy-array-without-
    scientific-notation-and-with-given-pre
    """
    std_options = _np.get_printoptions()
    _np.set_printoptions(*args, **kwargs)

    try:
        yield
    finally:
        _np.set_printoptions(**std_options)


def files_in_path(path: apt.PathType, ext: str = '.wav',
                  recursive: bool = False) -> apt.PathGen:
    """Generate all files with extension ``ext`` in ``path``.

    If `path` points to a file, it is yielded.
    If `path` points to a directory, all files with suffix `ext`
    are yielded.

    Args:
        path        (PathType) Path to audio file or folder of audio files.
        ext   (str)      The file type.
        recursive   (bool)     If True, recursively visite all subdirs of `path`.

    Yield:
        (PathGen)
    """
    path = pathlib.Path(path)

    if path.exists():
        if path.is_file():
            if path.suffix == ext:
                yield path.resolve()
            else:
                raise FileNotFoundError(f'File ``{path}`` is not a ``{ext}`` file.\n')
        else:
            name_pattern = '*'.join(ext)
            if recursive:
                for filepath in path.rglob(name_pattern):
                    yield filepath.resolve()
            else:
                for filepath in path.glob(name_pattern):
                    yield filepath.resolve()
    else:
        raise FileNotFoundError('Path "{}" could not be found.\n'.format(path))


def load(path: apt.PathType) -> typing.Any:
    """Load a pickled file.

    Args:
        path    (str) Path to file.

    Returns:
        (object) unpickled object
    """
    path = pathlib.Path(path)
    with path.open('rb') as file:
        data = pickle.load(file)
    return data

#TODO AttributeError if ext is None.
def repath(current_path: apt.PathType, new_path: apt.PathType,
           ext: apt.StrOrNone = None) -> apt.PathType:
    """Change the path and keep the file name. Optinally change the extension, too.

    Args:
        current_path (str or Path)  The path to change.
        new_path     (str or Path)  The new path.
        ext          (str or None)  Change file extension if ``ext`` is not None.

    Returns:
        (pathlib.Path)
    """
    current_path = pathlib.Path(current_path)
    new_path = pathlib.Path(new_path)

    if not ext.startswith('.'):
        ext = '.' + ext

    file_path = current_path.stem
    if ext is not None:
        file_path.join(ext)

    return new_path.joinpath(file_path)


def save(data: typing.Any, path: apt.PathType):
    """Pickles data to path.

    Args:
        data    (Any)         Pickleable object.
        path    (str or Path) Path to safe the file.
    """
    path = pathlib.Path(path)
    with path.open('wb') as file:
        pickle.dump(data, file)
