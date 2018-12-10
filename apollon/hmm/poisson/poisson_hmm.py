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

import json as _json
import pathlib as _pathlib
import typing as _typing
import warnings as _warning

import numpy as _np

from apollon.types import path_t
from apollon import tools as _tools
from apollon.hmm import utilities as _utils
from apollon.hmm.poisson.hmm_base import HMM_Base
import apollon.hmm.poisson.poisson_core as _core


arr_or_str_t = _typing.TypeVar('arr_or_str_t', str, _np.ndarray)
iter_or_none_t = _typing.TypeVar('iter_or_none_t', _typing.Iterable, None)
float_or_none_t = _typing.TypeVar('float_or_none_t', float, None)


class PoissonHMM(HMM_Base):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-statements

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['hyper_params', 'init_params', 'params',
                 'decoding' 'quality']

    def __init__(self, X: _np.ndarray, m: int,
                 init_lambda: arr_or_str_t = 'quantile',
                 init_gamma: arr_or_str_t = 'uniform',
                 init_delta: arr_or_str_t = 'stationary',
                 g_dirichlet: iter_or_none_t = None,
                 d_dirichlet: iter_or_none_t = None,
                 fill_diag: float_or_none_t = .8,
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

        self.hyper_params = HyperParameters(m, init_lambda, init_gamma, init_delta, g_dirichlet,
                                            d_dirichlet, fill_diag)

    def fit(self, X: _np.ndarray) -> bool:
        """Fit the initialized PoissonHMM to the input data set.

        Args:
            X   (np.ndarray)    Input data set.

        Returns:
            (int)   True on success else False.
        """
        check_poisson_input(X)

        if not em_ok:
            _warnings.warn('EM did not converge.', category=RuntimeWarning)
            return False
        return True


    def to_json(self, path):
        """Serialize HMM to JSON."""

    def to_txt(self, path):
        """Serialize HMM to text."""

    def to_pickle(self, path):
        """Serialize HMM to pickle."""




def to_txt(model: PoissonHMM, path: path_t):
    """Serialize `model` to `path` as text.

    Args:
        model   (PoissonHMM)    Any valid PoissonHMM instance.
        path    (str)           Save path.
    """
    path = _pathlib.Path(path)
    out_str = ('Name\n{}\n\n'
               + 'Training date\n{}\n\n'
               + 'Apollon version\n{}\n\n'
               + 'Initial Lambda\n{}\n\n'
               + 'Initial Delta\n{}\n\n'
               + 'Initial Gamma\n{}\n\n'
               + '\n------------------------------------------\n'
               + '\nLambda\n{}\n\n'
               + 'Delta\n{}\n\n'
               + 'Gamma\n{}\n\n'
               + '{:20}{:20}{:20}\n{:<20}{:<20}{:<20}\n')

    out_params = (path.stem,
                  model.training_date,
                  model.apollon_version,
                  model.init_lambda,
                  model.init_delta,
                  model.init_gamma,
                  model.lambda_,
                  model.delta_,
                  model.gamma_,
                  'nll', 'aic', 'aic',
                  model.nll, model.aic, model.bic)

    with path.open('w') as file:
        file.write(out_str.format(*out_params))


def to_json(model: PoissonHMM, path: path_t):
    """Serialize `model` to `path` using JSON.

    Args:
        model   (PoissonHMM)    Any valid PoissonHMM instance.
        path    (str)           Save path.
    """
    path = _pathlib.Path(path)
    data = {'name': path.stem,
            'training_date': model.training_date,
            'apollon_version': model.apollon_version,
            'init_lambda': model.init_lambda.tolist(),
            'init_delta':  model.init_delta.tolist(),
            'init_gamma':  model.init_gamma.tolist(),
            'lambda_': model.lambda_.tolist(),
            'delta_':  model.delta_.tolist(),
            'gamma_':  model.gamma_.tolist(),
            'nll': model.nll,
            'aic': model.aic,
            'bic': model.bic}

    with path.open('w') as file:
        _json.dump(data, file)


def to_pickle(model: PoissonHMM, path: path_t):
    """Serialize `model` to `path` using pickle.

    Args:
        model   (PoissonHMM)    Any valid PoissonHMM instance.
        path    (str)           Save path.
    """
    path = _pathlib.Path(path)
    with path.open('w') as file:
        file.dump(model, file)


class _HyperParameters:
    """Check and save model hyper parameters. Meant for compositional and internal only use.
    """
    __slots__ = ['m', 'valid_lambda_meths', 'valid_gamma_meths', 'valid_delta_meths',
                 'init_lambda_meth', 'init_gamma_meth', 'init_delta_meth',
                 'gamma_dp', 'delta_dp', 'fill_diag']

    def __init__(self, m: int,
                 init_lambda: arr_or_str_t,
                 init_gamma: arr_or_str_t,
                 init_delta: arr_or_str_t,
                 gamma_dp: iter_or_none_t = None,
                 delta_dp: iter_or_none_t = None,
                 fill_diag: float_or_none_t = None):
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
        self.valid_lambda_meths = ('linear', 'quantile', 'random')
        self.valid_gamma_meths = ('dirichlet', 'softmax', 'uniform')
        self.valid_delta_meths = ('dirichlet', 'softmax', 'stationary', 'uniform')

        if isinstance(m, int) and m > 0:
            self.m = m
        else:
            raise ValueError('Number of states `m` must be positiv integer.')

        self.gamma_dp = _tools.assert_and_pass(_assert_dirichlet_param, gamma_dp)
        self.delta_dp = _tools.assert_and_pass(_assert_dirichlet_param, delta_dp)
        self.fill_diag = _tools.assert_and_pass(_utils.assert_st_val, fill_diag)

        self.init_lambda_meth = self._assert_lambda(init_lambda)
        self.init_gamma_meth = self._assert_gamma(init_gamma, gamma_dp, fill_diag)
        self.init_delta_meth = self._assert_delta(init_delta, delta_dp)


    def _assert_lambda(self, _lambda: arr_or_str_t) -> _np.ndarray:
        """Assure that `_lambda` fits requirements for Poisson state-dependent means.

        Args:
            _lambda (str or np.ndarray)    Object to test.

        Returns:
            (np.ndarray)

        Raises:
            ValueError
            TypeError
        """
        if (isinstance(_lambda, str)):
            if _lambda not in self.valid_lambda_meths:
                raise ValueError('Unrecognized initialization method `{}`'.format(_lambda))

        elif isinstance(_lambda, _np.ndarray):
            _tools.assert_array(_lambda, 1, self.m, 0, name='init_lambda')

        else:
            raise TypeError(('Unrecognized type of param `init_lambda`.i Expected `str` or '
                             '`numpy.ndarray`, got {}.\n').format(type(_lambda)))
        return _lambda


    def _assert_gamma(self, _gamma: arr_or_str_t, gamma_dp: iter_or_none_t,
                      diag_val: float_or_none_t) -> _np.ndarray:
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

            if _gamma not in self.valid_gamma_meths:
                raise ValueError('Unrecognized initialization method `{}`'.format(_gamma))

            if _gamma == 'dirichlet' and gamma_dp is None:
                raise ValueError(('Hyper parameter `gamma_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `gamma`.'))

            if _gamma == 'uniform' and diag_vall is None:
                raise ValueError(('Hyper parameter `fill_diag` must be set when using initializer '
                                  '`uniform` for parameter `gamma`.'))

        elif isinstance(_gamma, _np.ndarray):
            _utils.assert_st_matrix(_gamma)

        else:
            raise TypeError(('Unrecognized type of argument `init_gamma`. Expected `str` or '
                             '`numpy.ndarray`, got {}.\n').format(type(_gamma)))
        return _gamma


    def _assert_delta(self, _delta: arr_or_str_t, delta_dp: iter_or_none_t) -> _np.ndarray:
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
        if (isinstance(_delta, str)):

            if _delta not in self.valid_delta_meths:
                raise ValueError('Unrecognized initialization method `{}`'.format(_delta))

            if _delta == 'dirichlet' and delta_dp is None:
                raise ValueError(('Hyper parameter `delta_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `delta`.'))

        elif isinstance(_delta, _np.ndarray):
            _utils.assert_st_vector(_delta)

        else:
            raise TypeError(('Unrecognized type of argument `init_delta`. Expected `str` or '
                             '`numpy.ndarray`, got {}.\n').format(type(delta)))
        return _delta


    def _assert_dirichlet_param(self, param: iter_or_none_t,
                                param_name: str = 'param') -> _np.ndarray:
        """Check for valid dirichlet params. On success cast `_param` to array and return.

        Args:
            param      (Iterable)    Parameter to check.
            param_name (str)         Parameter name.

        Returns:
            (np.ndarray)    Array of parameters.

        Raisese:
            ValueError
        """

        if param is None:
            return param

        param = _np.asarray(param)

        if param.size != self.m:
            raise ValueError('Size of `{}` must equal number of states.'.format(param_name))

        if _np.any(param < 0):
            raise ValueError('All elements of `{}` must be > 0.'.format(param_name))

        return param

