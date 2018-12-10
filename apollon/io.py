#!/usr/bin/python
# -*- coding: utf-8 -*-

"""apollon/IO.py

Tools for file access.

Classes:
    FileAccessControl       Descriptor for file name attributes.

Functions:
    files_in_folder         Iterate over all files in given folder.
    load                    Load pickled data.
    repath                  Change path but keep file name.
    save                    Pickle some data.
"""


__author__ = 'Michael Blaß'


import os as _os
import pathlib as _pathlib
import pickle as _pickle

from apollon import types as _types


class WavFileAccessControl:
    def __init__(self):
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


def files_in_path(path: _type.PathType, file_type:str = '.wav',
                  recursive:bool = False) -> _types.PathGen:
    """Generate all files with suffix `file_type` in `path`.

    If `path` points to a file, it is yielded. 
    If `path` points to a directory, all files with suffix `file_type`
    are yielded.

    Params:
        path        (path_t)   Path to audio file or folder of audio files.
        file_type   (str)      The file type.
        recursive   (bool)     If True, recursively visite all subdirs of `path`.
        
    Yield:
        (Path_Genrator_t)
    """
    path = _pathlib.Path(path)

    if path.exists():
        if path.is_file():
            if path.suffix == file_type:
                yield path.resolve()
            else:
                raise FileNotFoundError('File "{}" is not a "{}" file.\n'
                                        .format(path, file_type))
                
        else:              
            ft = '*' + file_type
            if recursive:
                for p in path.rglob(ft):
                    yield p.resolve()
            else:
                for p in path.glob(ft):
                    yield p.resolve()
    else:
        raise FileNotFoundError('Path "{}" could not be found.\n'.format(path))
        
        
def load(path):
    """Load a pickled file.

    Parameters:
        path    (str) Path to file.

    return      (object) unpickled object
    """
    with open(path, 'rb') as fobj:
        data = _pickle.load(fobj)
    return data


def repath(current_path, new_path, ext=None):
    """Change the path but keep the file name. Optinally chnage the
       extension, too.
    """
    current_path = _pathlib.Path(current_path)
    new_path = _pathlib.Path(new_path)

    if not ext.startswith('.'):
        ext = '.' + ext
    fn = current_path.stem if ext is None else current_path.stem + ext
    return new_path.joinpath(fn)


def save(data, path):
    """Pickles data to path.

    Parameters:
        data    (arbitray) pickleable object.
        path    (str) Path to safe the file.
    """
    with open(path, 'wb') as fobj:
        _pickle.dump(data, fobj)
