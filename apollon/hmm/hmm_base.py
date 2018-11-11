#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hmm_base.py -- Base class of the apollon Hidden Markov Model.
Copyright (C) 2017  Michael Blaß

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Classes:
    HMM_Base            Base class for all HMMs.
"""

import numpy as _np

import apollon
from . import utilities as _utils


class HMM_Base:
    """The Apollon Hidden Markov Model base class implements behaviour
       common for all types of HMMs.

       Coding convention:
           `_paramname` means an initial guess for a parameter.
           `paramname_` represents the estimated parameter.
    """

    def __init__(self, m, _gamma=None, _delta=None, verbose=True):

        if isinstance(m, int):
            self.m = m
        else:
            raise ValueError('Number of states must be integer.')

        self.verbose            = verbose
        self.trained            = False
        self.training_date      = ''
        self.apollon_version    = apollon.__version__



class PoissonHMM_Base(HMM_Base):
    def __init__(self, m,
                 _lambda='quantile', _gamma='uniform', _delta='stationary',
                 verbose=True, g_distr=None, d_distr=None, diag=.8):
        """Initialize PoissonHMM

        Args:
            m       (int)                   Number of states.
            _lambda (str or np.ndarray)     Initializer name of init values.
            _gamma  (str or np.ndarray)     Initializer name or init values.
            _delta  (str or np.ndarray)     Initializer name or init values.

            g_distr (iterable)  Dirichlet distribution params of len `m`.
                                Mandatory if `_gamma` == 'dirichlet'.

            d_distr (iterable)  Dirichlet distribution params of len `m`.
                                Mandatory if `delta` == 'dirichlet'.

            diag    (float)     Value on main diagonal of trans. prob matrix.
                                Mandatory if `_gamma`=='uniform'.
        """
        super().__init__(m, verbose=verbose)
        self._lambda = None
        self._gamma = None
        self._delta = None
        self.g_distr = g_distr
        self.d_distr = d_distr
        self.diag = diag

        # ---- check _lambda ----
        if isinstance(_lambda, str):

            if _lambda == 'linear':
                interface = '(x, self.m)'

            elif _lambda == 'quantile':
                interface = '(x, self.m)'

            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_lambda`.').format(_gamma))

            def _lambda_init_wrapper(method, interface):
                def __foo():
                    nonlocal self
                    return eval('_utils.init_lambda_{}{}'.format(method, interface))
                return __foo

            self._init_lambda = _lambda_init_wrapper(_lambda, interface)

        elif isinstance(_lambda, _np.ndarray):
            if _lambda.ndim != 1:
                raise ValueError(('Shape of param vector for `_lambda` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                  .format(_lambda.ndim))

            if _lambda.size != self.m:
                raise ValueError(('Number of params for `_lambda` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _lambda.size))

            def _lambda_init_wrapper(arr):
                def __foo():
                    return arr
                return __foo
            self._init_lambda = _lambda_init_wrapper(_lambda)

        else:
            raise TypeError(('Unrecognized type in param `_lambda`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_lambda)))

        # ---- check _gamma ---- 
        if isinstance(_gamma, str):

            if _gamma == 'dirichlet' and g_distr is not None:
                interface = '(self.m, self.g_distr)'

            elif _gamma == 'softmax':
                interface = '(self.m)'

            elif _gamma == 'uniform':
                interface = '(self.m, self.diag)'

            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_gamma`.').format(_gamma))

            def _gamma_init_wrapper(method, interface):
                def __foo():
                    nonlocal self
                    return eval('_utils.init_gamma_{}{}'.format(method, interface))
                return __foo

            self._init_gamma = _gamma_init_wrapper(_gamma, interface)

        elif isinstance(_gamma, _np.ndarray):
            if _gamma.ndim != 2:
                raise ValueError(('Shape of param vector for `_gamma` does '
                                  'not match HMM. Expected 2, got {}.\n')
                                  .format(_gamma.ndim))

            if _gamma.size != (self.m * self.m):
                raise ValueError(('Number of params for `_gamma` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _gamma.size))

            def _gamma_init_wrapper(arr):
                def __foo():
                    return arr
                return __foo
            self._init_gamma = _gamma_init_wrapper(_gamma)

        else:
            raise TypeError(('Unrecognized type in param `_gamma`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_gamma)))

        # ---- check _delta ----
        if isinstance(_delta, str):

            if _delta == 'dirichlet' and d_distr is not None:
                interface = '(self.m, self.d_distr)'

            elif _delta == 'softmax':
                interface = '(self.m)'

            elif _delta == 'stationary':
                interface = '(self._gamma)'

            elif _delta == 'uniform':
                interface = '(self.m)'

            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_delta`.').format(_delta))

            def _delta_init_wrapper(method, interface):
                def __foo():
                    nonlocal self
                    return eval('_utils.init_delta_{}{}'.format(method, interface))
                return __foo

            self._init_delta = _delta_init_wrapper(_delta, interface)

        elif isinstance(_delta, _np.ndarray):
            if _delta.ndim != 1:
                raise ValueError(('Shape of param vector for `_delta` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                  .format(_delta.ndim))

            if _delta.size != self.m:
                raise ValueError(('Number of params for `_delta` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _delta.size))

            def _delta_init_wrapper(arr):
                def __foo():
                    return arr
                return __foo
            self._init_delta = _delta_init_wrapper(_delta)

        else:
            raise TypeError(('Unrecognized type in param `_delta`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_delta)))

    def fit(self, x):
        self._lambda = self._init_lambda()
        self._gamma = self._init_gamma()
        self._delta = self._init_delta()

    def to_json(self, path):
        _utils.to_json(self, path)

    def to_txt(self, path):
        _utils.to_txt(self, path)


_lambda_init_methods = {
    'linear':
        {
            'method':       _utils.init_lambda_linear,
            'interface':    ('x', 'self.m')
        },

    'quantile':
        {
            'method':       _utils.init_lambda_quantile,
            'interface':    ('x', 'm')
        },

    'random':
        {
            'method':       _utils.init_lambda_random,
            'interface':    ('x', 'm')
        }
}

_gamma_init_methods = {
    'dirichlet':
        {
            'method':       _utils.init_gamma_dirichlet,
            'interface':    ('m', 'g_distr')
        },

    'softmax':
        {
            'method':       _utils.init_gamma_softmax,
            'interface':    ('m', )
        },

    'uniform':
        {
            'method':       _utils.init_gamma_uniform,
            'interface':    ('m', 'diag')
        }
}

_delta_init_methods = {
    'dirichlet':
        {
            'method':       _utils.init_delta_dirichlet,
            'interface':    ('m', 'd_distr')
        },

    'softmax':
        {
            'method':       _utils.init_delta_softmax,
            'interface':    ('m', )
        },

    'stationary':
        {
            'method':       _utils.init_delta_stationary,
            'interface':    ('self._gamma', )
        },

    'uniform':
        {
            'method':       _utils.init_delta_uniform,
            'interface':    ('m', )
        }
}
