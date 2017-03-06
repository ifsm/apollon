#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Michael Blaß'


class NotAudioDataError(Exception):
    '''Raised if a wrong data format is loaded by accident.
    '''
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ModelFailedError(Exception):
    '''Raised if HMM training failed.'''
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        return repr(self.value)
