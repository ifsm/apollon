#!/usr/bin/python
# -*- coding: utf-8 -*-

"""apollon/IO/tools.py

IO convenience functions.

Functions:
    files_in_folder    # iterate all files in given folder.
    save               # pickle some data
    load               # load pickled data
"""


__author__ = 'Michael Blaß'


import os as _os
import pathlib
import pickle as _pickle


def files_in_folder(path, suffix='.wav'):
    """Iterate over all file names in a given folder.

    Parameters:
        path        (str)  A path to a folder.
        file_type   (str)  Get only files with the given extension.
        abs_path    (bool) If True, yield absolute path of file.

    yield:    name of file.
    """
    wf = pathlib.Path(path)

    if wf.exists():
        wf_name = str(wf)

        if wf.is_dir():
            #verbose_msg('Accessing directory <{}> ...'.format(wf_name))
            for f in wf.iterdir():
                if f.is_file():
                    if not f.stem.startswith('.') and f.suffix == suffix:
                        yield f
        else:
            raise NotADirectoryError('<{}> is not a directory.\n'.format(wf_name))
    else:
        raise FileNotFoundError('File <{}> could not be found.\n'
                                .format(wf_name))


def files_in_folder_list(path, file_type='', abs_path=False):

    if not _os.path.exists(path):
        raise FileNotFoundError('Path {} does not exist.'.format(path))

    if not _os.path.isdir(path):
        raise NotADirectoryError('Path {} is not a directory.'.format(path))

    file_list = []
    for fname in _os.listdir(path):
        full_path = _os.path.join(path, fname)

        if fname.startswith('.'):
            continue

        if not _os.path.isfile(full_path):
            continue

        if fname.endswith(file_type):
            if abs_path:
                file_list.append(full_path)
            else:
                file_list.append(fname)

    return file_list


def load(path):
    """Load a pickled file.

    Parameters:
        path    (str) Path to file.

    return      (object) unpickled object
    """
    with open(path, 'rb') as fobj:
        data = _pickle.load(fobj)
    return data


def save(data, path):
    """Pickles data to path.

    Parameters:
        data    (arbitray) pickleable object.
        path    (str) Path to safe the file.
    """
    with open(path, 'wb') as fobj:
        _pickle.dump(data, fobj)
