#!/usr/bin/env python3

"""
poissonhmm.py -- HMM with Poisson-distributed state dependend process.
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
"""

"""
Functions:
    init_lambda_linear      Init linearly between min and max.
    init_lambda_quantile    Init regarding data quantiles.
    init_lambda_random      Init with random samples from data range.
    init_gamma_dirichlet    Init using Dirichlet distribution.
    init_gamma_softmax      Init with softmax of random floats.
    init_gamma_uniform      Init with uniform distr over the main diag.
    init_delta_dirichlet    Init using Dirichlet distribution.
    init_delta_softmax      Init with softmax of random floats.
    init_delta_stationary   Init with stationary distribution.
    init_delta_uniform      Init with uniform distribution.
    stationary_distr        Compute stationary distribution of tpm.
    to_txt                  Serializes model to text file.
    to_json                 JSON serialization.

    _is_tpm                 Check wheter array is stochastic matrix.
    _check_poisson_intput   Check wheter input is suitable for PoissonHMM.

Classes:
    PoissonHMM              HMM with univariat Poisson-distributed states.
"""

from typing import Iterable

import numpy as _np
from scipy import linalg as _linalg
from scipy import stats as _stats

from . hmm_base import HMM_Base
from . import poisson_core as core

class PoissonHMM(HMM_Base):

    __slots__ = ['m', '_init_args', 'apollon_version'
                 'lambda_', 'gamma_', 'delta_', 'theta',
                 'local_decoding', 'global_decoding',
                 'nll', 'aic', 'bic', 'training_date']

    def __init__(self, X, m,
                 _lambda='quantile', _gamma='uniform', _delta='stationary',
                 g_distr=None, d_distr=None, diag=.8, verbose=True):
        """Initialize PoissonHMM

        Args:
            x       (iterable of ints)      Data set.
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

        self._init_args = {
                '_lambda': _lambda,
                '_gamma' : _gamma,
                '_delta' : _delta,
                'g_distr': g_distr,
                'd_distr': d_distr,
                'diag'   : diag
            }


        # ---- initialize _lambda ----
        if isinstance(_lambda, str):
            if _lambda == 'linear':
                self._gamma = init_lambda_linear(X, self.m)
            elif _lambda == 'quantile':
                self._lambda = init_lambda_quantile(X, self.m)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_lambda`.').format(_lambda))

        elif isinstance(_lambda, _np.ndarray):
            if _lambda.ndim != 1:
                raise ValueError(('Shape of param vector for `_lambda` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                  .format(_lambda.ndim))

            if _lambda.size != self.m:
                raise ValueError(('Number of params for `_lambda` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _lambda.size))

            if _np.any(_lambda < 0.):
                raise ValueError(('Arguments for param `_lambda` must '
                                  'be positive.'))

            self._lambda = _lambda

        else:
            raise TypeError(('Unrecognized type in param `_lambda`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_lambda)))

        # ---- initialize _gamma ---- 
        if isinstance(_gamma, str):
            if _gamma == 'dirichlet' and g_distr is not None:
                self._gamma = init_gamma_dirichlet(m, g_distr)
            elif _gamma == 'softmax':
                self._gamma = init_gamma_softmax(m)
            elif _gamma == 'uniform':
                self._gamma = init_gamma_uniform(m, diag)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_gamma`.').format(_gamma))

        elif isinstance(_gamma, _np.ndarray):
            if _gamma.ndim != 2:
                raise ValueError(('Shape of param vector for `_gamma` does '
                                  'not match HMM. Expected 2, got {}.\n')
                                  .format(_gamma.ndim))

            if _gamma.size != (self.m * self.m):
                raise ValueError(('Number of params for `_gamma` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _gamma.size))

            if not _np.all(_np.isclose(_gamma.sum(axis=1), 1.)):
                raise ValueError(('Argument of paramter `_gamma` is not '
                                  'a valid stochastic matrix.'))
            self._gamma = _gamma

        else:
            raise TypeError(('Unrecognized type in param `_gamma`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_gamma)))

        # ---- initialize _delta ----
        if isinstance(_delta, str):
            if _delta == 'dirichlet' and d_distr is not None:
                self._delta = init_delta_dirichlet(self.m, d_distr)
            elif _delta == 'softmax':
                self._delta = init_delta_softmax(self.m)
            elif _delta == 'stationary':
                self._delta = init_delta_stationary(self._gamma)
            elif _delta == 'uniform':
                self._delta = init_delta_uniform(self.m)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `_delta`.').format(_delta))

        elif isinstance(_delta, _np.ndarray):
            if _delta.ndim != 1:
                raise ValueError(('Shape of param vector for `_delta` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                  .format(_delta.ndim))

            if _delta.size != self.m:
                raise ValueError(('Number of params for `_delta` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, _delta.size))

            if not _np.isclose(_delta.sum(), 1.):
                raise ValueError(('Argument for parameter `_delta` '
                                  'is not a valid stochastic vector.'))
            self._delta = _delta

        else:
            raise TypeError(('Unrecognized type in param `_delta`. '
                            'Expected `str` or `numpy.ndarray`, '
                            'got {}.\n').format(type(_delta)))

        self._theta = (self._lambda, self._gamma, self._delta)
        self.lambda_ = None
        self.gamma_ = None
        self.delta_ = None
        self.theta_ = None

    def fit(self, X):
        _check_poisson_input(X)


    def to_json(self, path):
        """Serialize HMM as JSON."""
        to_json(self, path)

    def to_txt(self, path):
        """Serialize HMM as text."""
        to_txt(self, path)

    def to_pickle(self, path):
        to_pickle(self, path):


def _check_poisson_input(X):
    """Check wheter data is a one-dimensional array of integer values.
    Otherwise raise an exception.
    """
    try:
        if X.ndim != 1:
            raise ValueError('Dimension of input vector must be 1.')
        if X.dtype.name != 'int64':
            raise TypeError('Input vector must be array of type int64')
    except AttributeError:
        raise AttributeError('Input vector must be numpy array')


def init_lambda_linear(x, m:int):
    """Linearily space m initial guesses in [min(data), max(data)].

        Params:
            x    (np.ndarray)   Input data.
            m    (int)          Number of states.

        Return:
            (np.ndarray)    m equidistant guesses.
    """
    return _np.linspace(min(x), max(x), m)


def init_lambda_quantile(x, m:int):
    """Compute m equally spaced percentiles from data.

    Params:
        x    (np.ndarray) Input data.
        m    (int)        Number of HMM states.

    Return:
        (np.ndarray)    m equally spaced percentiles.
    """
    if 3 <= m <= 100:
        pc = _np.linspace(100 / (m + 1), 100, m + 1)[:-1]
        return _np.percentile(x, pc)
    elif m == 2:
        return _np.percentile(x, [25, 75])
    elif m == 1:
        return _np.median(x)
    else:
        raise ValueError('Wrong input: m={}. 1 < m <= 100.'.format(m))


def init_lambda_random(x, m:int):
    """Init `_lambda` with random integers from [min(x), max(x)[.

    Params:
        x   (iterable)  Data set.
        m   (int)       Number of states.

    Retruns:
        (np.ndarray)    Initial guesses of shape (m, ).
    """
    return _np.random.randint(x.min(), x.max(), m).astype(float)


def init_gamma_dirichlet(m:int, alpha:Iterable):
    """
    Params:
        m       (int)       Number of states.
        alpha   (iterable)  Dirichlet distribution parameters.
                            Iterable of size m. Each entry controls
                            the probability mass that is put on the
                            respective transition.
    Returns:
        (np.ndarray)    Transition probability matrix of shape (m, m).
    """
    alpha = _np.atleast_1d(alpha)

    if alpha.ndim != 1:
        raise ValueError(('Wrong shape of param `alpha`. '
                         'Expected 1, got {}\n')
                         .format(alpha.ndim))

    if alpha.size != m:
        raise ValueError(('Wrong size of param `alpha`. '
                          'Expected {}, got {}\n')
                          .format(m, alpha.size))

    dv = (_stats.dirichlet(_np.roll(alpha, i)).rvs() for i in range(m))
    return _np.vstack(dv)


def init_gamma_softmax(m:int):
    """Initialize `_gamma` by applying softmax to a sample of random floats.

    Params:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Transition probability matrix of shape (m, m).
    """
    _gamma = _np.random.rand(m, m)
    return _np.exp(_gamma) / _np.exp(_gamma).sum(axis=1, keepdims=True)


def init_gamma_uniform(m:int, diag:float):
    """Fill the main diagonal of `_gamma` with `diag`. Set the
       off-diagoanl elements to the proportion of the remaining
       probability mass and the remaining number of elements per row.

        Params:
           m        (int)   Number of states.
           diag     (float) Value on main diagonal in [0, 1].

        Returns:
            (np.ndarray)    Transition probability matrix of shape (m, m).
    """
    if not isinstance(diag, float):
        raise TypeError(('Wrong type for param `diag`. '
                         'Expected <float>, got {}.\n')
                        .format(type(diag)))

    _gamma = _np.empty((m, m))
    _gamma.fill( (1-diag) / (m-1) )
    _np.fill_diagonal(_gamma, diag)

    return _gamma


def init_delta_dirichlet(m:int, alpha:Iterable):
    """
    Params:
        m       (int)       Number of states.
        alpha   (iterable)  Dirichlet distribution params.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    alpha = _np.atleast_1d(alpha)

    if alpha.ndim != 1:
        raise ValueError(('Wrong shape of param `alpha`. '
                         'Expected 1, got {}\n')
                         .format(alpha.ndim))

    if alpha.size != m:
        raise ValueError(('Wrong size of param `alpha`. '
                          'Expected {}, got {}\n')
                          .format(m, alpha.size))

    return _stats.dirichlet(alpha).rvs()


def init_delta_softmax(m:int):
    """Initialize `_delta` by applying softmax to a sample of random floats.

    Params:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    v = _np.random.rand(m)
    return _np.exp(v) / _np.exp(v).sum()


def init_delta_stationary(_gamma):
    """Initialize `_delta` with the stationary distribution of `_gamma`.

    Params:
        _gamma  (np.ndarray)    Initial transition probability matrix.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    return stationary_distr(_gamma)


def init_delta_uniform(m:int):
    """Initialize `_delta` with a uniform distribution.
    The initial values are set to the inverse of the number of states.

    Params:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    return _np.full(m, 1/m)


def stationary_distr(_gamma):
    """Calculate the stationary distribution of the transition probability
    matrix `_gamma`.

    Params:
        _gamma  (np.ndarray)    Transition probability matrix.

    Return:
        (np.ndarray)    Stationary distribution of shape (m, ).
    """
    if is_tpm(_gamma):
        m = _gamma.shape[0]
    else:
        raise ValueError('Matrix is not stochastic.')
    return _linalg.solve((_np.eye(m) - _gamma + 1).T, [1] * m)


def to_txt(model, path):
    path = pathlib.Path(path)
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
                  model._init_lambda,
                  model._init_delta,
                  model._init_gamma,
                  model.lambda_,
                  model.delta_,
                  model.gamma_,
                  'nll', 'aic', 'aic',
                  model.nll, model.aic, model.bic)

    with path.open('w') as file:
        file.write(out_str.format(*out_params))


def to_json(model, path):
    path = pathlib.Path(path)
    data = {'name': path.stem,
            'training_date': model.training_date,
            'apollon_version': model.apollon_version,
            '_init_lambda': model._init_lambda.tolist(),
            '_init_delta':  model._init_delta.tolist(),
            '_init_gamma':  model._init_gamma.tolist(),
            'lambda_': model.lambda_.tolist(),
            'delta_':  model.delta_.tolist(),
            'gamma_':  model.gamma_.tolist(),
            'nll': model.nll,
            'aic': model.aic,
            'bic': model.bic}

    with path.open('w') as file:
        json.dump(data, file)


def to_pickle(model, path):
    path = pathlib.Path(path)
    with path.open('w') as file:
        file.dump(data, file)


def is_tpm(arr):
    """Test wheter `arr` is a valid stochastic matrix.

    A stochastic matrix is a (1) two-dimensional, (2) quadratic
    matrix, whose row all sum up to exactly 1.

    Params:
        arr (np.ndarray)    Input array.

    Returns:
        True

    Raises:
        ValueError
    """
    if arr.ndim != 2:
        raise ValueError('Matrix must be two-dimensional.')

    if arr.shape[0] != arr.shape[1]:
        raise ValueError('Matrix must be quadratic.')

    if not _np.all(_np.isclose(arr.sum(axis=1), 1.)):
        raise ValueError('Matrix is not row-stochastic.')

    return True
