#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""hmm_base.py

(c) Michael Blaß 2016

Base class and untility functions Hidden Morkov Models.

Classes:
    HMM_Base            Base class for all HMMs.
"""

import abc as _abc
import numpy as _np

from . import utilities as _utils


class HMM_Base(metaclass=_abc.ABCMeta):
    def __init__(self, x, m, init_gamma=None, init_delta=None,
                 guess='quantile', verbose=True):
        # Flags
        self.verbose = verbose
        self.trained = False

        self.x = _np.atleast_1d(x)
        if self.x.ndim > 1:
            raise ValueError('Input array must be 1d.')

        if isinstance(m, int):
            self.m = m
        else:
            raise ValueError('Number of states must be integer.')

        # set initial transition probability matrix
        self._init_gamma = _utils.new_gamma(m) \
            if init_gamma is None else init_gamma

        # calculate stationary distribution
        self._init_delta = _utils.calculate_delta(self._init_gamma) \
            if init_delta is None else init_delta

    @_abc.abstractmethod
    def natural_params(self, w_params):
        '''Transform working parameters to natural params.

            Params:
                w_params    (np.ndarray) of working params.

            Return:
                distr_param_1, ..., distr_param_n, gamma, delta.'''
        pass

    @_abc.abstractmethod
    def working_params(self, _gamma, *args):
        '''Transform natural params to working params.

            Params:
                _gamma    (np.ndarray) Transition probability matrix.
                *args     each (np.ndarray) distr_param_1, ..., distr_param_n

            Return:
                (np.ndarray)    Working params.'''
        pass

    @_abc.abstractmethod
    def sample(self, n: int) -> _np.ndarray :
        '''Draw samples form HMM.

        Params:
            n    (int) Number of samples to draw.

        Return:
            (np.ndarray) of length n holding samples.
        '''
        pass
