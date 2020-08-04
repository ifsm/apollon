"""
poisson_hmm.py -- HMM with Poisson-distributed state dependent process.
Copyright (C) 2018  Michael Blaß <mblass@posteo.net>

Functions:
    to_txt                  Serializes model to text file.
    to_json                 JSON serialization.

    is_tpm                 Check weter array is stochastic matrix.
    _check_poisson_intput   Check wheter input is suitable for PoissonHMM.

Classes:
    PoissonHMM              HMM with univariat Poisson-distributed states.
"""

import typing as _typing
import warnings as _warnings

import numpy as _np

import chainsaddiction as _ca

import apollon
from apollon import types as _at
import apollon.io.io as aio
from apollon.types import Array as _Array
from apollon import tools as _tools
import apollon.hmm.utilities as ahu


class PoissonHmm:

    # pylint: disable=too-many-arguments

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['hyper_params', 'init_params', 'params', 'decoding', 'quality',
                 'verbose', 'version', 'training_date', 'success']

    def __init__(self, X: _Array, m_states: int,
                 init_lambda: _at.ArrayOrStr = 'quantile',
                 init_gamma: _at.ArrayOrStr = 'uniform',
                 init_delta: _at.ArrayOrStr = 'stationary',
                 g_dirichlet: _at.IterOrNone = None,
                 d_dirichlet: _at.IterOrNone = None,
                 fill_diag: float = .8,
                 verbose: bool = True):

        """Initialize PoissonHMM

        Args:
            X           (np.ndarray of ints)    Data set.
            m_states    (int)                   Number of states.
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
        self.training_date = _tools.time_stamp()
        self.verbose = verbose
        self.version = apollon.__version__
        self.hyper_params = _HyperParams(m_states, init_lambda, init_gamma, init_delta,
                                         g_dirichlet, d_dirichlet, fill_diag)

        self.init_params = _InitParams(X, self.hyper_params)
        self.params = None
        self.quality = None
        self.success = None

    def fit(self, X: _Array) -> bool:
        """Fit the initialized PoissonHMM to the input data set.

        Args:
            X   (np.ndarray)    Input data set.

        Returns:
            (int)   True on success else False.
        """
        assert_poisson_input_data(X)

        res = _ca.hmm_poisson_fit_em(X, self.hyper_params.m_states,
                                     *self.init_params.__dict__.values(), 1000, 1e-5)

        self.success = True if res[0] == 1 else False
        self.params = Params(*res[1:4])
        self.quality = QualityMeasures(*res[4:])

        if self.success is False:
            _warnings.warn('EM did not converge.', category=RuntimeWarning)


    def score(self, X: _Array):
        """Compute the log-likelihood of `X` under this HMM."""

    def to_dict(self):
        """Returns HMM parameters as dict."""
        attrs = ('hyper_params', 'init_params', 'params',
                 'quality', 'success')
        out = {}
        for key in attrs:
            try:
                out[key] = getattr(self, key).__dict__
            except AttributeError:
                out[key] = getattr(self, key)
        return out


class _HyperParams:

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    """Check and save model hyper parameters. Meant for compositional and internal only use.
    """

    def __init__(self,
                 m_states: int,
                 init_lambda: _at.ArrayOrStr,
                 init_gamma: _at.ArrayOrStr,
                 init_delta: _at.ArrayOrStr,
                 gamma_dp: _at.IterOrNone = None,
                 delta_dp: _at.IterOrNone = None,
                 fill_diag: float = None):
        """Check and save model hyper parameters.

        Args:
            m_states        (int)
            init_lambda     (str or ndarray)
            init_gamma      (str or ndarray)
            init_delta      (str or ndarray)
            gamma_dp        (tuple)
            delta_dp        (tuple)
            fill_diag       (float)
        """

        if isinstance(m_states, int) and m_states > 0:
            self.m_states = m_states
        else:
            raise ValueError('Number of states `m` must be positiv integer.')

        self.gamma_dp = _tools.assert_and_pass(self._assert_dirichlet_param, gamma_dp)
        self.delta_dp = _tools.assert_and_pass(self._assert_dirichlet_param, delta_dp)
        self.fill_diag = _tools.assert_and_pass(ahu.assert_st_val, fill_diag)

        self.init_lambda_meth = self._assert_lambda(init_lambda)
        self.init_gamma_meth = self._assert_gamma(init_gamma, gamma_dp, fill_diag)
        self.init_delta_meth = self._assert_delta(init_delta, delta_dp)


    def to_dict(self):
        """Return dict representation."""
        return {attr: getattr(self, attr) for attr in self.__slots__}


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
            if _lambda not in ahu.StateDependentMeansInitializer.methods:
                raise ValueError('Unrecognized initialization method `{}`'.format(_lambda))

        elif isinstance(_lambda, _np.ndarray):
            _tools.assert_array(_lambda, 1, self.m_states, 0, name='init_lambda')

        else:
            raise TypeError(('Unrecognized type of param ``init_lambda`` Expected ``str`` or '
                             '``numpy.ndarray``, got {}.\n').format(type(_lambda)))
        return _lambda

    @staticmethod
    def _assert_gamma(_gamma: _at.ArrayOrStr, gamma_dp: _at.IterOrNone,
                      diag_val: float) -> _np.ndarray:
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

            if _gamma not in ahu.TpmInitializer.methods:
                raise ValueError('Unrecognized initialization method `{}`'.format(_gamma))

            if _gamma == 'dirichlet' and gamma_dp is None:
                raise ValueError(('Hyper parameter `gamma_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `gamma`.'))

            if _gamma == 'uniform' and diag_val is None:
                raise ValueError(('Hyper parameter `fill_diag` must be set when using initializer '
                                  '`uniform` for parameter `gamma`.'))

        elif isinstance(_gamma, _np.ndarray):
            ahu.assert_st_matrix(_gamma)
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

            if _delta not in ahu.StartDistributionInitializer.methods:
                raise ValueError('Unrecognized initialization method `{}`'.format(_delta))

            if _delta == 'dirichlet' and delta_dp is None:
                raise ValueError(('Hyper parameter `delta_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `delta`.'))

        elif isinstance(_delta, _np.ndarray):
            ahu.assert_st_vector(_delta)

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

        if param.size != self.m_states:
            raise ValueError('Size of dirichlet parameter must equal number of states.')

        if _np.any(param < 0):
            raise ValueError('All elements of dirichlet parameter must be > 0.')


    def __str__(self):
        return str(list(self.__dict__.items()))

    def __repr__(self):
        items = ('\t{}={!r}'.format(name, attr)
                 for name, attr in self.__dict__.items())
        return '_HyerParameters(\n{})'.format(',\n'.join(items))




class _InitParams:
    """Initialize PoissonHmm parameters.
    """
    def __init__(self, X: _np.ndarray, hy_params: _HyperParams):
        """
        """

        assert_poisson_input_data(X)
        self.lambda_ = self._init_lambda(hy_params, X)
        self.gamma_ = self._init_gamma(hy_params)
        self.delta_ = self._init_delta(hy_params)


    @staticmethod
    def _init_lambda(hy_params: _HyperParams, X: _Array) -> _Array:
        if isinstance(hy_params.init_lambda_meth, _np.ndarray):
            return hy_params.init_lambda_meth.copy()

        if hy_params.init_lambda_meth == 'hist':
            return ahu.StateDependentMeansInitializer.hist(X, hy_params.m_states)

        if hy_params.init_lambda_meth == 'linear':
            return ahu.StateDependentMeansInitializer.linear(X, hy_params.m_states)

        if hy_params.init_lambda_meth == 'quantile':
            return ahu.StateDependentMeansInitializer.quantile(X, hy_params.m_states)

        if hy_params.init_lambda_meth == 'random':
            return ahu.StateDependentMeansInitializer.random(X, hy_params.m_states)

        raise ValueError("Unknown init method or init_lambda_meth is not an array.")


    @staticmethod
    def _init_gamma(hy_params: _HyperParams) -> _Array:

        if isinstance(hy_params.init_gamma_meth, _np.ndarray):
            return hy_params.init_gamma_meth.copy()

        if hy_params.init_gamma_meth == 'dirichlet':
            return ahu.TpmInitializer.dirichlet(hy_params.m_states, hy_params.gamma_dp)

        if hy_params.init_gamma_meth == 'softmax':
            return ahu.TpmInitializer.softmax(hy_params.m_states)

        if hy_params.init_gamma_meth == 'uniform':
            return ahu.TpmInitializer.uniform(hy_params.m_states, hy_params.fill_diag)

        raise ValueError("Unknown init method or init_gamma_meth is not an array.")


    def _init_delta(self, hy_params: _HyperParams) -> _Array:
        if isinstance(hy_params.init_delta_meth, _np.ndarray):
            return hy_params.init_delta_meth.copy()

        if hy_params.init_delta_meth == 'dirichlet':
            return ahu.StartDistributionInitializer.dirichlet(hy_params.m_states,
                                                                 hy_params.delta_dp)

        if hy_params.init_delta_meth == 'softmax':
            return ahu.StartDistributionInitializer.softmax(hy_params.m_states)

        if hy_params.init_delta_meth == 'stationary':
            return ahu.StartDistributionInitializer.stationary(self.gamma_)

        if hy_params.init_delta_meth == 'uniform':
            return ahu.StartDistributionInitializer.uniform(hy_params.m_states)

        raise ValueError("Unknown init method or init_delta_meth is not an array.")

    def __str__(self):
        with aio.array_print_opt(precision=4, suppress=True):
            out = 'Initial Lambda:\n{}\n\nInitial Gamma:\n{}\n\nInitial Delta:\n{}\n'
            out = out.format(*self.__dict__.values())
        return out

    def __repr__(self):
        return self.__str__()


class QualityMeasures:
    """
    """
    def __init__(self, aic, bic, nll, n_iter):
        self.aic = aic
        self.bic = bic
        self.nll = nll
        self.n_iter = n_iter

    def __str__(self):
        return 'AIC = {}\nBIC = {}\nNLL = {}\nn_iter = {}'.format(*self.__dict__.values())

    def __repr__(self):
        return self.__str__()


class Params:
    """Easy access to estimated HMM parameters and quality measures.
    """
    def __init__(self, lambda_, gamma_, delta_):
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.delta_ = delta_

    def __str__(self):
        with aio.array_print_opt(precision=4, suppress=True):
            out = 'Lambda:\n{}\n\nGamma:\n{}\n\nDelta:\n{}\n'
            out = out.format(*self.__dict__.values())
        return out

    def __repr__(self):
        return self.__str__()


def assert_poisson_input_data(X: _np.ndarray):
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
