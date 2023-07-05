"""
HMM with Poisson-distributed state dependent process
"""

import typing as _typing
from typing import Any
import warnings as _warnings

import numpy as _np

from chainsaddiction import poishmm

import apollon
from apollon import types as _at
from apollon.io.utils import array_print_opt
from apollon.types import FloatArray, IntArray
from apollon import tools as _tools
import apollon.hmm.utilities as ahu


class PoissonHmm:

    # pylint: disable=too-many-arguments

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['hyper_params', 'init_params', 'params', 'decoding', 'quality',
                 'verbose', 'version', 'training_date', 'success']

    def __init__(self, data: IntArray, m_states: int,
                 init_lambda: _at.ArrayOrStr = 'quantile',
                 init_gamma: _at.ArrayOrStr = 'uniform',
                 init_delta: _at.ArrayOrStr = 'stationary',
                 g_dirichlet: _at.IterOrNone = None,
                 d_dirichlet: _at.IterOrNone = None,
                 fill_diag: float = .8,
                 verbose: bool = True):

        """Initialize PoissonHMM

        Args:
            data: Data set
            m_states: Number of model states
            init_lambda: Method name or array of init values
            init_gamma: Method name or array of init values
            init_delta: Method name or array of init values

            gamma_dp: Dirichlet distribution params of len `m`.
                      Mandatory if `init_gamma` == 'dirichlet'.

            delta_dp: Dirichlet distribution params of len `m`
                      Mandatory if `delta` == 'dirichlet'.

            fill_diag: Value on main diagonal of transition probability matrix.
                       Mandatory if `init_gamma` == 'uniform'.
        """
        self.training_date = _tools.time_stamp()
        self.verbose = verbose
        self.version = apollon.__version__
        self.hyper_params = _HyperParams(m_states, init_lambda, init_gamma, init_delta,
                                         g_dirichlet, d_dirichlet, fill_diag)

        self.init_params = _InitParams(data, self.hyper_params)
        self.params: Params
        self.quality: QualityMeasures
        self.success: bool

    def fit(self, data: IntArray) -> bool:
        """Fit the initialized PoissonHMM to the input data set.

        Args:
            data: Input data

        Returns:
            `True` on success else `False`
        """
        assert_poisson_input_data(data)

        res = poishmm.fit(data.size, self.hyper_params.m_states, 1000, 1e-5,
                          self.init_params.lambda_, self.init_params.gamma_,
                          self.init_params.delta_, data)

        self.success = not res.err
        self.params = Params(res.lambda_.astype(_np.float64),
                             res.gamma_.astype(_np.float64),
                             res.delta_.astype(_np.float64))
        self.quality = QualityMeasures(res.aic, res.bic, res.llk, res.n_iter)

        if self.success is False:
            _warnings.warn('EM did not converge.', category=RuntimeWarning)
        return self.success


    def score(self, data: IntArray) -> float:
        """Compute the log-likelihood of `data` under this HMM.

        Args:
            data: Input data

        Returns:
            Log-likelihood of the data under the model parameters."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
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

    __slots__ = ["m_states", "gamma_dp", "delta_dp", "fill_diag", "init_lambda_meth",
                 "init_gamma_meth", "init_delta_meth"]

    def __init__(self,
                 m_states: int,
                 init_lambda: _at.ArrayOrStr,
                 init_gamma: _at.ArrayOrStr,
                 init_delta: _at.ArrayOrStr,
                 gamma_dp: _at.IterOrNone = None,
                 delta_dp: _at.IterOrNone = None,
                 fill_diag: float | None = None) -> None:
        """Check and save model hyper parameters.

        Args:
            m_states:    Number of model states
            init_lambda: Method name or array of init values
            init_gamma:  Method name or array of init values
            init_delta:  Method name or array of init values

            gamma_dp: Dirichlet distribution params of len `m`.
                      Mandatory if `init_gamma` == 'dirichlet'.

            delta_dp: Dirichlet distribution params of len `m`
                      Mandatory if `delta` == 'dirichlet'.

            fill_diag: Value on main diagonal of transition probability matrix.
                       Mandatory if `init_gamma` == 'uniform'.
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


    def to_dict(self) -> dict[str, Any]:
        """Return dict representation."""
        return {attr: getattr(self, attr) for attr in self.__slots__}


    def _assert_lambda(self, _lambda: FloatArray | str) -> FloatArray | str:
        """Assure that `_lambda` fits requirements for Poisson state-dependent means.

        Args:
            _lambda: Array of state dependent means

        Returns:
            Unchanged ``_lambda``, if all requirements are met.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_lambda, str):
            if _lambda not in ahu.StateDependentMeansInitializer.methods:
                raise ValueError(f'Unrecognized initialization method `{_lambda}`')

        elif isinstance(_lambda, _np.ndarray):
            _tools.assert_array(_lambda, 1, self.m_states, 0, name='init_lambda')

        else:
            raise TypeError(f'Unrecognized type of param ``init_lambda``. '
                             f'Expected ``str`` or ``numpy.ndarray``, got  '
                             f'{type(_lambda)}.')
        return _lambda

    @staticmethod
    def _assert_gamma(_gamma: FloatArray | str, gamma_dp: _at.IterOrNone,
                      diag_val: float | None) -> FloatArray | str:
        """Assure that `_gamma` fits requirements for Poisson transition probability matirces.

        Args:
            _gamma: Transition probability matrix
            _gamma_dp: Dirichlet distribution parameters
            _fill_val: Fill value for main diagonal

        Returns:
            ``_gamma unchanged, if all requirements are met.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_gamma, str):

            if _gamma not in ahu.TpmInitializer.methods:
                raise ValueError(f'Unrecognized initialization method `{_gamma}`')

            if _gamma == 'dirichlet' and gamma_dp is None:
                raise ValueError('Hyper parameter `gamma_dp` must be set when using initializer '
                                  '`dirichlet` for parameter `gamma`.')

            if _gamma == 'uniform' and diag_val is None:
                raise ValueError('Hyper parameter `fill_diag` must be set when using initializer '
                                  '`uniform` for parameter `gamma`.')

        elif isinstance(_gamma, _np.ndarray):
            ahu.assert_st_matrix(_gamma)
        else:
            raise TypeError('Unrecognized type of argument `init_gamma`. '
                            'Expected `str` or `numpy.ndarray`, '
                            f'got {type(_gamma)}.')
        return _gamma

    @staticmethod
    def _assert_delta(_delta: FloatArray | str, delta_dp: _at.IterOrNone) -> FloatArray | str:
        """Assure that `_delta` fits requirements for Poisson initial distributions.

        Args:
            _delta: Array of start distribution
            delta_dp: Dirichlet distribution params

        Returns:
            `_delta` unchanged, if all requirements are met.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(_delta, str):

            if _delta not in ahu.StartDistributionInitializer.methods:
                raise ValueError(f'Unrecognized initialization method `{_delta}`')

            if _delta == 'dirichlet' and delta_dp is None:
                raise ValueError('Hyper parameter `delta_dp` must be set when '
                                 'using initializer `dirichlet` for parameter `delta`.')

        elif isinstance(_delta, _np.ndarray):
            ahu.assert_st_vector(_delta)

        else:
            raise TypeError('Unrecognized type of argument `init_delta`. '
                            'Expected `str` or `numpy.ndarray`, got {type(_delta)}.')
        return _delta


    def _assert_dirichlet_param(self, param: _typing.Iterable[float]) -> None:
        """Check for valid dirichlet params.

        Dirichlet parameter vectors are iterables of positive floats. Their
        len must equal to the given number of states.

        Args:
            param: Dirichlet distribution parameters

        Raises:
            ValueError
        """
        param = _np.asarray(param).astype(_np.float64)

        if param.size != self.m_states:
            raise ValueError('Size of dirichlet parameter must equal number of states.')

        if _np.any(param < 0):
            raise ValueError('All elements of dirichlet parameter must be > 0.')


    def __str__(self) -> str:
        return str(list(self.__dict__.items()))

    def __repr__(self) -> str:
        items = (f'\t{name}={attr!r}'
                 for name, attr in self.__dict__.items())
        return f'_HyerParameters(\n{",".join(items)})'


class _InitParams:
    """Initialize PoissonHmm parameters.
    """
    def __init__(self, data: IntArray, hy_params: _HyperParams) -> None:
        """Create initial parameters.

        Args:
            data: Input data
            hy_params: Hyper parameters
        """
        assert_poisson_input_data(data)
        self.lambda_ = self._init_lambda(hy_params, data)
        self.gamma_ = self._init_gamma(hy_params)
        self.delta_ = self._init_delta(hy_params)


    @staticmethod
    def _init_lambda(hy_params: _HyperParams, data: IntArray) -> FloatArray:
        if isinstance(hy_params.init_lambda_meth, _np.ndarray):
            return hy_params.init_lambda_meth.copy()

        if hy_params.init_lambda_meth == 'hist':
            return ahu.StateDependentMeansInitializer.hist(data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'linear':
            return ahu.StateDependentMeansInitializer.linear(data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'quantile':
            return ahu.StateDependentMeansInitializer.quantile(data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'random':
            return ahu.StateDependentMeansInitializer.random(data, hy_params.m_states)

        raise ValueError("Unknown init method or init_lambda_meth is not an array.")


    @staticmethod
    def _init_gamma(hy_params: _HyperParams) -> FloatArray:

        if isinstance(hy_params.init_gamma_meth, _np.ndarray):
            return hy_params.init_gamma_meth.copy()

        if hy_params.init_gamma_meth == 'dirichlet':
            return ahu.TpmInitializer.dirichlet(hy_params.m_states, hy_params.gamma_dp)

        if hy_params.init_gamma_meth == 'softmax':
            return ahu.TpmInitializer.softmax(hy_params.m_states)

        if hy_params.init_gamma_meth == 'uniform':
            return ahu.TpmInitializer.uniform(hy_params.m_states, hy_params.fill_diag)

        raise ValueError("Unknown init method or init_gamma_meth is not an array.")


    def _init_delta(self, hy_params: _HyperParams) -> FloatArray:
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

    def __str__(self) -> str:
        with array_print_opt(precision=4, suppress=True):
            out = 'Initial Lambda:\n{}\n\nInitial Gamma:\n{}\n\nInitial Delta:\n{}\n'
            out = out.format(*self.__dict__.values())
        return out

    def __repr__(self) -> str:
        return self.__str__()


class QualityMeasures:
    """
    """
    def __init__(self, aic: float, bic: float, nllk: float, n_iter: int) -> None:
        self.aic = aic
        self.bic = bic
        self.nllk = nllk
        self.n_iter = n_iter

    def __str__(self) -> str:
        tmp = 'AIC = {}\nBIC = {}\nNLL = {}\nn_iter = {}'
        return tmp.format(*self.__dict__.values())

    def __repr__(self) -> str:
        return self.__str__()


class Params:
    """Easy access to estimated HMM parameters and quality measures.
    """
    def __init__(self, lambda_: FloatArray, gamma_: FloatArray, delta_: FloatArray) -> None:
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.delta_ = delta_

    def __str__(self) -> str:
        with array_print_opt(precision=4, suppress=True):
            out = 'Lambda:\n{}\n\nGamma:\n{}\n\nDelta:\n{}\n'
            out = out.format(*self.__dict__.values())
        return out

    def __repr__(self) -> str:
        return self.__str__()


def assert_poisson_input_data(data: IntArray) -> None:
    """Raise if data is not a array of integers.

    Args:
        data: Input array

    Raises:
        ValueError
    """
    if not isinstance(data, _np.ndarray):
        raise TypeError('Data set is not a numpy array.')

    if data.dtype is not _np.dtype(_np.int64):
        raise TypeError('Elements of input data set must be integers.')

    if _np.any(data < 0):
        raise ValueError('Elements of input data must be positive.')
