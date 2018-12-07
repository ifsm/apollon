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

    is_tpm                 Check wheter array is stochastic matrix.
    _check_poisson_intput   Check wheter input is suitable for PoissonHMM.

Classes:
    PoissonHMM              HMM with univariat Poisson-distributed states.
"""

import json
from dataclasses import dataclass
import pathlib
import typing
import warnings

import numpy as _np
from scipy import linalg as _linalg
from scipy import stats as _stats

from apollon.types import path_t
from apollon.hmm.hmm_base import HMM_Base
import apollon.hmm.poisson_core as _core


class PoissonHMM(HMM_Base):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-statements

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['m', '_init_args', 'apollon_version'
                 'lambda_', 'gamma_', 'delta_', 'theta',
                 'local_decoding', 'global_decoding',
                 'nll', 'aic', 'bic', 'training_date']

    def __init__(self, X, m,
                 init_lambda='quantile', init_gamma='uniform', init_delta='stationary',
                 g_distr=None, d_distr=None, diag=.8, verbose=True):
        """Initialize PoissonHMM

        Args:
            x            (iterable of ints)      Data set.
            m            (int)                   Number of states.
            init_lambda (str or np.ndarray)     Initializer name of init values.
            init_gamma  (str or np.ndarray)     Initializer name or init values.
            init_delta       (str or np.ndarray)     Initializer name or init values.

            g_distr (iterable)  Dirichlet distribution params of len `m`.
                                Mandatory if `init_gamma` == 'dirichlet'.

            d_distr (iterable)  Dirichlet distribution params of len `m`.
                                Mandatory if `delta` == 'dirichlet'.

            diag    (float)     Value on main diagonal of trans. prob matrix.
                                Mandatory if `init_gamma`=='uniform'.
        """
        super().__init__(m, verbose=verbose)

        self._hyper_params = {
            'm': m,
            'init_lambda_meth': init_lambda,
            'init_gamma_meth' : init_gamma,
            'init_delta_meth' : init_delta,
            'g_distr': g_distr,
            'd_distr': d_distr,
            'diag'   : diag
        }

        # TODO Check shape of `g_distr` and `d_distr` here.

        # ---- initialize init_lambda ----
        if isinstance(init_lambda, str):
            if init_lambda == 'linear':
                self.init_lambda = init_lambda_linear(X, self.m)
            elif init_lambda == 'quantile':
                self.init_lambda = init_lambda_quantile(X, self.m)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `init_lambda`.').format(init_lambda))

        elif isinstance(init_lambda, _np.ndarray):
            if init_lambda.ndim != 1:
                raise ValueError(('Shape of param vector for `init_lambda` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                 .format(init_lambda.ndim))

            if init_lambda.size != self.m:
                raise ValueError(('Number of params for `init_lambda` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, init_lambda.size))

            if _np.any(init_lambda < 0.):
                raise ValueError(('Arguments for param `init_lambda` must '
                                  'be positive.'))

            self.init_lambda = init_lambda

        else:
            raise TypeError(('Unrecognized type in param `init_lambda`. '
                             'Expected `str` or `numpy.ndarray`, '
                             'got {}.\n').format(type(init_lambda)))

        # ---- initialize init_gamma ----
        if isinstance(init_gamma, str):
            if init_gamma == 'dirichlet' and g_distr is not None:
                self.init_gamma = init_gamma_dirichlet(m, g_distr)
            elif init_gamma == 'softmax':
                self.init_gamma = init_gamma_softmax(m)
            elif init_gamma == 'uniform':
                self.init_gamma = init_gamma_uniform(m, diag)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `init_gamma`.').format(init_gamma))

        elif isinstance(init_gamma, _np.ndarray):
            if init_gamma.ndim != 2:
                raise ValueError(('Shape of param vector for `init_gamma` does '
                                  'not match HMM. Expected 2, got {}.\n')
                                 .format(init_gamma.ndim))

            if init_gamma.size != (self.m * self.m):
                raise ValueError(('Number of params for `init_gamma` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, init_gamma.size))

            if not _np.all(_np.isclose(init_gamma.sum(axis=1), 1.)):
                raise ValueError(('Argument of paramter `init_gamma` is not '
                                  'a valid stochastic matrix.'))
            self.init_gamma = init_gamma

        else:
            raise TypeError(('Unrecognized type in param `init_gamma`. '
                             'Expected `str` or `numpy.ndarray`, '
                             'got {}.\n').format(type(init_gamma)))

        # ---- initialize init_delta ----
        if isinstance(init_delta, str):
            if init_delta == 'dirichlet' and d_distr is not None:
                self.init_delta = init_delta_dirichlet(self.m, d_distr)
            elif init_delta == 'softmax':
                self.init_delta = init_delta_softmax(self.m)
            elif init_delta == 'stationary':
                self.init_delta = init_delta_stationary(self.init_gamma)
            elif init_delta == 'uniform':
                self.init_delta = init_delta_uniform(self.m)
            else:
                raise KeyError(('Unknown initialization method `{}` '
                                'for param `init_delta`.').format(init_delta))

        elif isinstance(init_delta, _np.ndarray):
            if init_delta.ndim != 1:
                raise ValueError(('Shape of param vector for `init_delta` does '
                                  'not match HMM. Expected 1, got {}.\n')
                                 .format(init_delta.ndim))

            if init_delta.size != self.m:
                raise ValueError(('Number of params for `init_delta` does not '
                                  'match number of HMM states. Expected, '
                                  '{}, got {}.\n').format(self.m, init_delta.size))

            if not _np.isclose(init_delta.sum(), 1.0):
                raise ValueError(('Argument for parameter `init_delta` '
                                  'is not a valid stochastic vector.'))
            self.init_delta = init_delta

        else:
            raise TypeError(('Unrecognized type in param `init_delta`. '
                             'Expected `str` or `numpy.ndarray`, '
                             'got {}.\n').format(type(init_delta)))

        self._theta = (self.init_lambda, self.init_gamma, self.init_delta)
        self.lambda_ = None
        self.gamma_ = None
        self.delta_ = None
        self.theta_ = None

    def fit(self, X: _np.ndarray) -> bool:
        """Fit the initialized PoissonHMM to the input data set.

        Args:
            X   (np.ndarray)    Input data set.

        Returns:
            (int)   True on success else False.
        """
        check_poisson_input(X)

        self.theta_, self.nll, em_ok = _core.poisson_EM(X, self.m, self._theta)
        self.lambda_, self.gamma_, self.delta_ = self.theta_
        self.aic = self.compute_aic()
        self.bic = self.compute_bic(X.size)

        self.delta_ = stationary_distr(self.gamma_)
        self.viterbi = _core.poisson_viterbi(self, X)

        if not em_ok:
            warnings.warn('EM did not converge.', category=RuntimeWarning)

        return True

    def to_json(self, path):
        """Serialize HMM as JSON."""
        to_json(self, path)

    def to_txt(self, path):
        """Serialize HMM as text."""
        to_txt(self, path)

    def to_pickle(self, path):
        """Serialzed HMM to pickle."""

def check_poisson_input(X):
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


def init_lambda_linear(X: _np.ndarray, m: int) -> _np.ndarray:
    """Initialize state-dependent means with `m` linearily spaced values
    from ]min(data), max(data)[.

        Args:
            X    (np.ndarray)   Input data.
            m    (int)          Number of states.

        Returns:
            (np.ndarray)    Initial state-dependent means of shape (m, ).
    """
    bordered_space = _np.linspace(X.min(), X.max(), m+2)
    return bordered_space[1:-1]


def init_lambda_quantile(X: _np.ndarray, m: int) -> _np.ndarray:
    """Initialize state-dependent means with `m` equally spaced
    percentiles from data.

    Args:
        X    (np.ndarray) Input data.
        m    (int)        Number of HMM states.

    Returns:
        (np.ndarray)    Initial state-dependent means of shape (m, ).
    """
    if 3 <= m <= 100:
        q_range = _np.linspace(100 / (m + 1), 100, m + 1)[:-1]
        return _np.percentile(X, q_range)

    if m == 2:
        return _np.percentile(X, [25, 75])

    if m == 1:
        return _np.median(X)

    raise ValueError('Wrong input: m={}. 1 < m <= 100.'.format(m))


def init_lambda_random(X: _np.ndarray, m: int) -> _np.ndarray:
    """Initialize state-dependent means with random integers from
    [min(x), max(x)[.

    Args:
        X   (np.ndarray)    Data set.
        m   (int)           Number of states.

    Retruns:
        (np.ndarray)    Initial state-dependent means of shape (m, ).
    """
    return _np.random.randint(X.min(), X.max(), m).astype(float)


def init_gamma_dirichlet(m: int, alpha: tuple) -> _np.ndarray:
    """
    Args:
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

    distr = (_stats.dirichlet(_np.roll(alpha, i)).rvs() for i in range(m))
    return _np.vstack(distr)


def init_gamma_softmax(m: int) -> _np.ndarray:
    """Initialize `init_gamma` by applying softmax to a sample of random floats.

    Args:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Transition probability matrix of shape (m, m).
    """
    init_gamma = _np.random.rand(m, m)
    return _np.exp(init_gamma) / _np.exp(init_gamma).sum(axis=1, keepdims=True)


def init_gamma_uniform(m: int, diag: float) -> _np.ndarray:
    """Fill the main diagonal of `init_gamma` with `diag`. Set the
       off-diagoanl elements to the proportion of the remaining
       probability mass and the remaining number of elements per row.

        Args:
           m        (int)   Number of states.
           diag     (float) Value on main diagonal in [0, 1].

        Returns:
            (np.ndarray)    Transition probability matrix of shape (m, m).
    """
    if not isinstance(diag, float):
        raise TypeError(('Wrong type for param `diag`. '
                         'Expected <float>, got {}.\n')
                        .format(type(diag)))

    init_gamma = _np.empty((m, m))
    init_gamma.fill((1-diag) / (m-1))
    _np.fill_diagonal(init_gamma, diag)

    return init_gamma


def init_delta_dirichlet(m: int, alpha: tuple) -> _np.ndarray:
    """Initialize the initial distribution with a Dirichlet random sample.

    Args:
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


def init_delta_softmax(m: int) -> _np.ndarray:
    """Initialize the initial distribution by applying softmax to a sample
    of random floats.

    Args:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    rnd_vals = _np.random.rand(m)
    return _np.exp(rnd_vals) / _np.exp(rnd_vals).sum()


def init_delta_stationary(gamma_: _np.ndarray) -> _np.ndarray:
    """Initialize the initial distribution with the stationary
    distribution of `init_gamma`.

    Args:
        gamma_  (np.ndarray)    Initial transition probability matrix.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    return stationary_distr(gamma_)


def init_delta_uniform(m: int) -> _np.ndarray:
    """Initialize the initial distribution uniformly.
    The initial values are set to the inverse of the number of states.

    Args:
        m   (int)   Number of states.

    Returns:
        (np.ndarray)    Stochastic vector of shape (m, ).
    """
    return _np.full(m, 1/m)


def stationary_distr(tpm: _np.ndarray) -> _np.ndarray:
    """Calculate the stationary distribution of the transition probability
    matrix `tpm`.

    Args:
        tpm (np.ndarray)    Transition probability matrix.

    Returns:
        (np.ndarray)    Stationary distribution of shape (m, ).
    """
    if is_tpm(tpm):
        m = tpm.shape[0]
    else:
        raise ValueError('Matrix is not stochastic.')
    return _linalg.solve((_np.eye(m) - tpm + 1).T, _np.ones(m))


def to_txt(model: PoissonHMM, path: path_t):
    """Serialize `model` to `path` as text.

    Args:
        model   (PoissonHMM)    Any valid PoissonHMM instance.
        path    (str)           Save path.
    """
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
    path = pathlib.Path(path)
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
        json.dump(data, file)


def to_pickle(model: PoissonHMM, path: path_t):
    """Serialize `model` to `path` using pickle.

    Args:
        model   (PoissonHMM)    Any valid PoissonHMM instance.
        path    (str)           Save path.
    """
    path = pathlib.Path(path)
    with path.open('w') as file:
        file.dump(model, file)


def is_tpm(arr: _np.ndarray):
    """Test wheter `arr` is a valid stochastic matrix.

    A stochastic matrix is a (1) two-dimensional, (2) quadratic
    matrix, whose (3) rows all sum up to exactly 1.

    Args:
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

Initializer = typing.TypeVar('Initializer', str, _np.ndarray)

@dataclass
class HyperParams:
    init_lambda_meth: Initializer
    init_gamma_meth: Initializer
    init_delta_meth: Initializer
    g_dirichlet: tuple
    d_dirichlet: tuple
    fill_diag: float


