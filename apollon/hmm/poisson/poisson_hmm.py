"""
poisson_hmm.py -- HMM with Poisson-distributed state dependent process.
Copyright (C) 2018  Michael Blaß <michael.blass@uni-hamburg.de>

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


Functions:
    to_txt                  Serializes model to text file.
    to_json                 JSON serialization.

    is_tpm                 Check weter array is stochastic matrix.
    _check_poisson_intput   Check wheter input is suitable for PoissonHMM.

Classes:
    PoissonHMM              HMM with univariat Poisson-distributed states.
"""

#import json as _json
#import pathlib as _pathlib
import typing as _typing
import warnings as _warnings

import numpy as _np

from apollon import types as _at
from apollon import tools as _tools
from apollon.hmm import hmm_utilities as _utils
from apollon.hmm.poisson.poisson_base import HMM_Base
#import apollon.hmm.poisson.poisson_core as _core


VALID_LAMBDA_METHS = ('linear', 'quantile', 'random')
VALID_GAMMA_METHS = ('dirichlet', 'softmax', 'uniform')
VALID_DELTA_METHS = ('dirichlet', 'softmax', 'stationary', 'uniform')


class PoissonHMM(HMM_Base):
    # pylint: disable=too-many-arguments

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['hyper_params', 'init_params', 'params', 'decoding', 'quality']

    def __init__(self, X: _np.ndarray, m: int,
                 init_lambda: _at.ArrayOrStr = 'quantile',
                 init_gamma: _at.ArrayOrStr = 'uniform',
                 init_delta: _at.ArrayOrStr = 'stationary',
                 g_dirichlet: _at.IterOrNone = None,
                 d_dirichlet: _at.IterOrNone = None,
                 fill_diag: _at.FloatOrNone = .8,
                 verbose: bool = True):

        """Initialize PoissonHMM

        Args:
            X           (np.ndarray of ints)    Data set.
            m           (int)                   Number of states.
            init_lambda (str or np.ndarray)     Method name or array of init values.
            init_gamma  (str or np.ndarray)     Method name or array of init values.
            init_delta  (str or np.ndarray)     Method name or array of init values.

            gamma_dp    (iterable or None)  Dirichlet distribution params of len `m`.
                                            Mandatory if `init_gamma` == 'dirichlet'.

            delta_dp    (iterable or None)  Dirichlet distribution params of len `m`.
                                            Mandatory if `delta` == 'dirichlet'.

            fill_diag   (float or None)     Value on main diagonal of tran sition prob matrix.
                                            Mandatory if `init_gamma` == 'uniform'.
        """
        super().__init__(m, verbose)

        self.hyper_params = _HyperParameters(m, init_lambda, init_gamma, init_delta, g_dirichlet,
                                             d_dirichlet, fill_diag)

    def fit(self, X: _np.ndarray) -> bool:
        """Fit the initialized PoissonHMM to the input data set.

        Args:
            X   (np.ndarray)    Input data set.

        Returns:
            (int)   True on success else False.
        """
        assert_poisson_input(X)
        em_ok = True
        if not em_ok:
            _warnings.warn('EM did not converge.', category=RuntimeWarning)


class _HyperParameters:

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    """Check and save model hyper parameters. Meant for compositional and internal only use.
    """
    __slots__ = ['m', 'init_lambda_meth', 'init_gamma_meth', 'init_delta_meth', 'gamma_dp',
                 'delta_dp', 'fill_diag']

    def __init__(self,
                 m: int,
                 init_lambda: _at.ArrayOrStr,
                 init_gamma: _at.ArrayOrStr,
                 init_delta: _at.ArrayOrStr,
                 gamma_dp: _at.IterOrNone = None,
                 delta_dp: _at.IterOrNone = None,
                 fill_diag: _at.FloatOrNone = None):
        """Check and save model hyper parameters.

        Args:
            m              (int)
            init_lambda    (str or ndarray)
            init_gamma     (str or ndarray)
            init_delta     (str or ndarray)
            gamma_dp       (tuple)
            delta_dp       (tuple)
            fill_diag      (float)
        """

        if isinstance(m, int) and m > 0:
            self.m = m
        else:
            raise ValueError('Number of states `m` must be positiv integer.')

        self.gamma_dp = _tools.assert_and_pass(self._assert_dirichlet_param, gamma_dp)
        self.delta_dp = _tools.assert_and_pass(self._assert_dirichlet_param, delta_dp)
        self.fill_diag = _tools.assert_and_pass(_utils.assert_st_val, fill_diag)

        self.init_lambda_meth = self._assert_lambda(init_lambda)
        self.init_gamma_meth = self._assert_gamma(init_gamma, gamma_dp, fill_diag)
        self.init_delta_meth = self._assert_delta(init_delta, delta_dp)

    def _assert_lambda(self, _lambda: _at.ArrayOrStr) -> _np.ndarray:
        """Assure that `_lambda` fits requirements for Poisson state-dependent means.

        Args:
            _lambda (str or np.ndarray)    Object to test.

        Returns:
            (np.ndarray)

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_lambda, str):
            if _lambda not in VALID_LAMBDA_METHS:
                raise ValueError('Unrecognized initialization method `{}`'.format(_lambda))

        elif isinstance(_lambda, _np.ndarray):
            _tools.assert_array(_lambda, 1, self.m, 0, name='init_lambda')

        else:
            raise TypeError(('Unrecognized type of param ``init_lambda`` Expected ``str`` or '
                             '``numpy.ndarray``, got {}.\n').format(type(_lambda)))
        return _lambda

    @staticmethod
    def _assert_gamma(_gamma: _at.ArrayOrStr, gamma_dp: _at.IterOrNone,
                      diag_val: _at.FloatOrNone) -> _np.ndarray:
        """Assure that `_gamma` fits requirements for Poisson transition probability matirces.

        Args:
            _gamma    (str or np.ndarray)    Object to test.
            _gamma_dp (Iterable or None)     Dirichlet params.
            _fill_val (float or None)        Fill value for main diagonal.

        Returns:
            (np.ndarray)

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_gamma, str):

            if _gamma not in VALID_GAMMA_METHS:
                raise ValueError('Unrecognized initialization method `{}`'.format(_gamma))

            if _gamma == 'dirichlet' and gamma_dp is None:
                raise ValueError(('Hyper parameter `gamma_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `gamma`.'))

            if _gamma == 'uniform' and diag_val is None:
                raise ValueError(('Hyper parameter `fill_diag` must be set when using initializer '
                                  '`uniform` for parameter `gamma`.'))

        elif isinstance(_gamma, _np.ndarray):
            _utils.assert_st_matrix(_gamma)
        else:
            raise TypeError(('Unrecognized type of argument `init_gamma`. Expected `str` or '
                             '`numpy.ndarray`, got {}.\n').format(type(_gamma)))
        return _gamma

    @staticmethod
    def _assert_delta(_delta: _at.ArrayOrStr, delta_dp: _at.IterOrNone) -> _np.ndarray:
        """Assure that `_delta` fits requirements for Poisson initial distributions.

        Args:
            _delta   (str or np.ndarray)    Object to test.
            delta_dp (Iterable)             Dirichlet params.

        Returns:
            (np.ndarray)

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_delta, str):

            if _delta not in VALID_DELTA_METHS:
                raise ValueError('Unrecognized initialization method `{}`'.format(_delta))

            if _delta == 'dirichlet' and delta_dp is None:
                raise ValueError(('Hyper parameter `delta_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `delta`.'))

        elif isinstance(_delta, _np.ndarray):
            _utils.assert_st_vector(_delta)

        else:
            raise TypeError(('Unrecognized type of argument `init_delta`. Expected `str` or '
                             '`numpy.ndarray`, got {}.\n').format(type(_delta)))
        return _delta


    def _assert_dirichlet_param(self, param: _typing.Iterable) -> _np.ndarray:
        """Check for valid dirichlet params.

        Dirichlet parameter vectors are iterables of positive floats. Their
        len must equal to the given number of states.

        Args:
            param      (Iterable)    Parameter to check.

        Raises:
            ValueError
        """
        param = _np.asarray(param)

        if param.size != self.m:
            raise ValueError('Size of dirichlet parameter must equal number of states.')

        if _np.any(param < 0):
            raise ValueError('All elements of dirichlet parameter must be > 0.')


    def __str__(self):
        vals = []
        for slot in self.__slots__:
            vals.append(getattr(self, slot))
        return str(vals)

    def __repr__(self):
        args = ''
        for attr in self.__slots__:
            args += '\n\t{attr} = {val},'.format(attr=attr, val=repr(getattr(self, attr)))
        return '_HyerParameters({})'.format(args.strip(','))


def assert_poisson_input(X: _np.ndarray):
    """Raise if X is not a array of integers.

    Args:
        X (np.ndarray)    Data set.

    Raises:
        ValueError
    """
    if not isinstance(X, _np.ndarray):
        raise TypeError('Data set is not a numpy array.')

    if X.dtype is not _np.dtype(_np.int64):
        raise TypeError('Elements of input data set must be integers.')

    if _np.any(X < 0):
        raise ValueError('Elements of input data must be positive.')
