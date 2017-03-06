#!/usr/bin/python
# -*- coding: utf-8 -*-


"""apollon/IO/file_properties.py

Definition of file properties

Classes:
    _FileProperties
"""

import os as _os


class _FileProperties:
    '''Extracts and keeps file name and absolute path from input string.

    Methods:
        get_path        Return absolute path
        get_fname       Return file name
    '''
    def __init__(self, *args):
        if len(args) == 0:
            self._path = None
            self._fname = None

        elif len(args) == 1:
            self._path = _os.path.dirname(_os.path.abspath(args[0]))
            self._fname = _os.path.basename(_os.path.abspath(args[0]))

        else:
            self._path = args[0]
            self._fname = args[1]

    def get_path(self):
        return self._path

    def get_fname(self):
        return self._fname
