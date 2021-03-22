"""
:copyright: 2018, Michael Blaß
:license: BSD 3 Clause

.. autosummary::
    :nosignatures:

    PoissonHMM
    to_txt
    to_json
    is_tpm
    _check_poisson_intput
"""


import typing as _typing
import warnings as _warnings

import numpy as _np
import chainsaddiction as _ca

import apollon
from apollon.types import Array, ArrayOrStr, IterOrNone
import apollon.io.io as aio
from apollon import tools as _tools
import apollon.hmm.utilities as ahu


class PoissonHmm:

    # pylint: disable=too-many-arguments

    """Hidden-Markov Model with univariate Poisson-distributed states."""

    __slots__ = ['hyper_params', 'init_params', 'params', 'decoding', 'quality',
                 'verbose', 'version', 'training_date', 'success']

    def __init__(self, train_data: Array, m_states: int,
                 init_lambda: ArrayOrStr = 'quantile',
                 init_gamma: ArrayOrStr = 'uniform',
                 init_delta: ArrayOrStr = 'stationary',
                 g_dirichlet: IterOrNone = None,
                 d_dirichlet: IterOrNone = None,
                 fill_diag: float = .8,
                 verbose: bool = True):

        """Initialize PoissonHMM

        Args:
            train_data:     Data set.
            m_states:       Number of states.
            init_lambda:    Method name or array of init values.
            init_gamma:     Method name or array of init values.
            init_delta:     Method name or array of init values.

            gamma_dp:       Dirichlet distribution params of len ``m``.
                            Mandatory if ``init_gamma`` == 'dirichlet'.

            delta_dp:       Dirichlet distribution params of len ``m``.
                            Mandatory if ``delta`` == 'dirichlet'.

            fill_diag:      Value on main diagonal of tran sition prob matrix.
                            Mandatory if `init_gamma` == 'uniform'.
        """
        self.training_date = _tools.time_stamp()
        self.verbose = verbose
        self.version = apollon.__version__
        self.hyper_params = _HyperParams(m_states, init_lambda, init_gamma, init_delta,
                                         g_dirichlet, d_dirichlet, fill_diag)

        self.init_params = _InitParams(train_data, self.hyper_params)
        self.params = None
        self.quality = None
        self.success = None


    def fit(self, train_data: Array) -> None:
        """Fit the initialized PoissonHMM to the input data.

        Args:
            train_data: Input data set.
        """
        assert_poisson_input_data(train_data)

        res = _ca.hmm_poisson_fit_em(train_data, self.hyper_params.m_states,
                                     *self.init_params.__dict__.values(), 1000, 1e-5)

        self.success = True if res[0] == 1 else False
        self.params = Params(*res[1:4])
        self.quality = QualityMeasures(*res[4:])

        if self.success is False:
            _warnings.warn('EM did not converge.', category=RuntimeWarning)


    def score(self, train_data: Array):
        """Compute the log-likelihood of `train_data` under this HMM."""
        pass


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

    def __init__(self, m_states: int, init_lambda: ArrayOrStr,
                 init_gamma: ArrayOrStr, init_delta: ArrayOrStr,
                 gamma_dp: IterOrNone = None, delta_dp: IterOrNone = None,
                 fill_diag: float = None) -> None:
        """Check and save model hyper parameters.

        Args:
            m_states:       Number of states.
            init_lambda:    Method name or array of init values.
            init_gamma:     Method name or array of init values.
            init_delta:     Method name or array of init values.
            gamma_dp:       Distribution params for tpm.
            delta_dp:       Distribution params for delta.
            fill_diag:      Fill value for tpm main diagonal.
        """
        if isinstance(m_states, int) and m_states > 0:
            self.m_states = m_states
        else:
            raise ValueError(('Number of states `m_states` must be positiv '
                             'integer.'))

        self.gamma_dp = _tools.assert_and_pass(self._assert_dirichlet_param, gamma_dp)
        self.delta_dp = _tools.assert_and_pass(self._assert_dirichlet_param, delta_dp)
        self.fill_diag = _tools.assert_and_pass(ahu.assert_st_val, fill_diag)

        self.init_lambda_meth = self._assert_lambda(init_lambda)
        self.init_gamma_meth = self._assert_gamma(init_gamma, gamma_dp, fill_diag)
        self.init_delta_meth = self._assert_delta(init_delta, delta_dp)


    def to_dict(self):
        """Return dict representation."""
        return {attr: getattr(self, attr) for attr in self.__slots__}


    def _assert_lambda(self, lambda_: ArrayOrStr) -> Array:
        """Assure that ``lambda_`` fits requirements for Poisson state-dependent means.

        Args:
            lambda_:    Object to test.

        Returns:
            `lambda_`, if tests pass.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(lambda_, str):
            if lambda_ not in ahu.StateDependentMeansInitializer.methods:
                raise ValueError('Unrecognized initialization method `{}`'.format(lambda_))

        elif isinstance(lambda_, _np.ndarray):
            _tools.assert_array(lambda_, 1, self.m_states, 0, name='init_lambda')

        else:
            raise TypeError(('Unrecognized type of param ``init_lambda`` Expected ``str`` or '
                             '``numpy.ndarray``, got {}.\n').format(type(lambda_)))
        return lambda_


    @staticmethod
    def _assert_gamma(gamma_: ArrayOrStr, ddp: IterOrNone,
                      diag_val: float) -> Array:
        """Assure that ``gamma_`` fits requirements for Poisson transition
        probability matirces.

        Args:
            gamma_:     Transition probability matrix.
            ddp:        Optional params for Dirichlet initialization.
            fill_val:  Fill value for main diagonal.

        Returns:
            ``gamma``, iff all assertions pass.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(gamma_, str):

            if gamma_ not in ahu.TpmInitializer.methods:
                msg = f'Unrecognized initialization method `{gamma_}`.'
                raise ValueError(msg)

            if gamma_ == 'dirichlet' and gamma_dp is None:
                msg = ('Hyper parameter `gamma_dp` must be set when using '
                       'initializer `dirichlet` for parameter `gamma`.')
                raise ValueError(msg)

            if gamma_ == 'uniform' and diag_val is None:
                msg = ('Hyper parameter `fill_diag` must be set when using '
                       'initializer `uniform` for parameter `gamma`.')
                raise ValueError(msg)

        elif isinstance(gamma_, _np.ndarray):
            ahu.assert_st_matrix(gamma_)
        else:
            msg = (f'Unrecognized type of argument `init_gamma`. Expected '
                   '`str` or `numpy.ndarray`, got {gamma_}.\n')
            raise TypeError(msg)
        return gamma_


    @staticmethod
    def _assert_delta(delta_: ArrayOrStr, ddp: IterOrNone) -> Array:
        """Assure that ``delta_`` fits requirements for Poisson initial
        distributions.

        Args:
            delta_: Initial distribution.
            ddp:    Optional params for Dirichlet initialization.

        Returns:
            ``delta_``, iff all assertions pass.

        Raises:
            ValueError
            TypeError
        """
        if isinstance(delta_, str):

            if delta_ not in ahu.StartDistributionInitializer.methods:
                msg =(f'Unrecognized initialization method `{delta_}`.'
                raise ValueError()

            if delta_ == 'dirichlet' and ddp is None:
                msg = ('Hyper parameter `ddp` not set, but is required when
                       'using initializer `dirichlet` for parameter `delta`.')
                raise ValueError(msg)

        elif isinstance(delta_, _np.ndarray):
            ahu.assert_st_vector(delta_)

        else:
            msg = (f'Unrecognized type of argument `init_delta`. Expected '
                   f'`str` or `numpy.ndarray`, got {delta_}.')
            raise TypeError(msg)
        return delta_


    def _assert_dirichlet_param(self, ddp: _typing.Iterable) -> Array:
        """Check for valid Dirichlet params.

        Dirichlet parameter vectors are iterables of positive floats. Their
        length must equal the given number of states.

        Args:
            ddp:    Dirichlet distribution params.

        Raises:
            ValueError
        """
        ddp = _np.asarray(ddp)

        if ddp.size != self.m_states:
            msg = (f'Length of `ddp` ({ddp}) not equal to number of states '
                  f'({self.m_states}).')
            raise ValueError(msg)

        if _np.any(ddp < 0):
            msg = 'Dirichelt param less than 0.'
            raise ValueError(msg)


    def __str__(self):
        return str(list(self.__dict__.items()))

    def __repr__(self):
        items = ('\t{}={!r}'.format(name, attr)
                 for name, attr in self.__dict__.items())
        return '_HyerParameters(\n{})'.format(',\n'.join(items))




class _InitParams:
    """Initialize PoissonHmm parameters.
    """
    def __init__(self, train_data: Array, hy_params: _HyperParams):
        """
        """

        assert_poisson_input_data(train_data)
        self.lambda_ = self._init_lambda(hy_params, train_data)
        self.gamma_ = self._init_gamma(hy_params)
        self.delta_ = self._init_delta(hy_params)


    @staticmethod
    def _init_lambda(hy_params: _HyperParams, train_data: Array) -> Array:
        if isinstance(hy_params.init_lambda_meth, _np.ndarray):
            return hy_params.init_lambda_meth.copy()

        if hy_params.init_lambda_meth == 'hist':
            return ahu.StateDependentMeansInitializer.hist(train_data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'linear':
            return ahu.StateDependentMeansInitializer.linear(train_data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'quantile':
            return ahu.StateDependentMeansInitializer.quantile(train_data, hy_params.m_states)

        if hy_params.init_lambda_meth == 'random':
            return ahu.StateDependentMeansInitializer.random(train_data, hy_params.m_states)

        msg = 'Unknown init method or init_lambda_meth is not an array.'
        raise ValueError(msg)


    @staticmethod
    def _init_gamma(hy_params: _HyperParams) -> Array:

        if isinstance(hy_params.init_gamma_meth, _np.ndarray):
            return hy_params.init_gamma_meth.copy()

        if hy_params.init_gamma_meth == 'dirichlet':
            return ahu.TpmInitializer.dirichlet(hy_params.m_states, hy_params.gamma_dp)

        if hy_params.init_gamma_meth == 'softmax':
            return ahu.TpmInitializer.softmax(hy_params.m_states)

        if hy_params.init_gamma_meth == 'uniform':
            return ahu.TpmInitializer.uniform(hy_params.m_states, hy_params.fill_diag)

        msg = 'Unknown init method or init_gamma_meth is not an array.'
        raise ValueError(msg)


    def _init_delta(self, hy_params: _HyperParams) -> Array:
        if isinstance(hy_params.init_delta_meth, _np.ndarray):
            return hy_params.init_delta_meth.copy()

        if hy_params.init_delta_meth == 'dirichlet':
            return ahu.StartDistributionInitializer.dirichlet(hy_params.m_states, hy_params.delta_dp)

        if hy_params.init_delta_meth == 'softmax':
            return ahu.StartDistributionInitializer.softmax(hy_params.m_states)

        if hy_params.init_delta_meth == 'stationary':
            return ahu.StartDistributionInitializer.stationary(self.gamma_)

        if hy_params.init_delta_meth == 'uniform':
            return ahu.StartDistributionInitializer.uniform(hy_params.m_states)

        msg = 'Unknown init method or init_delta_meth is not an array.'
        raise ValueError(msg)


    def __str__(self):
        with aio.array_print_opt(precision=4, suppress=True):
            temp = ('Initial Lambda:\n{}\n\n'
                    'Initial Gamma:\n{}\n\n'
                    'Initial Delta:\n{}\n')
            return = temp.format(*self.__dict__.values())


    def __repr__(self):
        return self.__str__()


class QualityMeasures:
    """Dataclass to encapsulate HMM quality measures."""
    def __init__(self, aic: float, bic: float, nll: float,
                 n_iter: int) -> None:
        self.aic = aic
        self.bic = bic
        self.nll = nll
        self.n_iter = n_iter

    def __str__(self):
        temp ='AIC = {}\nBIC = {}\nNLL = {}\nn_iter = {}'
        return temp.format(*self.__dict__.values())

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


def assert_poisson_input_data(train_data: Array) -> None:
    """Raise if train_data is not a array of integers.

    Args:
        train_data: Data set.

    Raises:
        ValueError
    """
    if not isinstance(train_data, _np.ndarray):
        raise TypeError('Data set is not a numpy array.')

    if train_data.dtype is not _np.dtype(_np.int64):
        raise TypeError('Elements of input data set must be integers.')

    if _np.any(train_data < 0):
        raise ValueError('Found negative values in `train_data`.')
