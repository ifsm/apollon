#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""apollon/hmm/poisson_hmm.py

(c) Michael Blaß 2016


A simple implementation of a PoissonHMM including all helper functions.

Classes:
    PoissonHmm          Fitting PoissonHmm to given data.

Functions:
    hmm_distance        Calculate dissimilarity of two PoissonHmms.
    LL                  Calculate log-likelihood of PoissonHMM.
    sample              Sample from PoissonHMM.
"""


from collections import deque
import sys

import numpy as _np
from numpy import linalg as _linalg
from scipy import stats as _stats
from scipy.optimize import minimize, fmin_powell, approx_fprime
import warnings as _warnings

from apollon import aplot
from apollon import _defaults
from apollon import exceptions as _except
from apollon.hmm.hmm_base import HMM_Base as _HMM_Base
from apollon.hmm import utilities as _utils
from apollon import tools as _tools
from . import poisson_core as _poisson_core


class PoissonHmm(_HMM_Base):

    __slots__ = ['x', 'm', 'params',
                 '_init_lambda', '_init_gamma', '_init_delta',
                 'lambda_', 'gamma_', 'delta_', 'theta',
                 'llk', 'aic', 'bic',
                 'distr_params', 'model_params',
                 'local_decoding', 'global_decoding',
                 '_support']

    # TODO: Add docstring
    def __init__(self, x, m, init_lambda=None, init_gamma=None,
                 init_delta=None, guess='quantile', verbose=True):

        super().__init__(x, m, init_gamma, init_delta,
                         guess='quantile', verbose=verbose)
        self._support = _np.arange(0, self.x.max(), dtype=int)

        # TODO: Params should be NamedTuple
        self.distr_params = ['lambda_']
        self.model_params = ['m', 'gamma_', 'delta_']

        # Initialize distriution parameter
        if guess in _defaults.lambda_guess_methods:
            if guess == 'linear':
                self._init_lambda = _utils.guess_linear(self.x, m) \
                    if init_lambda is None else init_lambda
            elif guess == 'quantile':
                self._init_lambda = _utils.guess_qunatile(self.x, m) \
                    if init_lambda is None else init_lambda
        else:
            raise ValueError('Method <{}> not supported.'.format(guess))

        self.local_decoding = None
        self.global_decoding = None


    def __str__(self):
        if self.trained:
            out_str = ('Lambda\n{}\n\nDelta\n{}\n\nGamma\n{}\n\n' +
                       '{:15}{:15}{:15}\n{:<15}{:<15}{:<15}')

            out_params = (self.lambda_.round(4), self.delta_.round(4),
                          self.gamma_.round(4), 'Mllk', 'AIC', 'BIC',
                          self.llk.round(4), self.aic.round(4),
                          self.bic.round(4))
        else:
            out_str = 'init_Lambda\n{}\n\ninit_Delta\n{}\n\ninit_Gamma\n{}'
            out_params = (self._init_lambda.round(2),
                          self._init_delta.round(2),
                          self._init_gamma.round(2))

        return out_str.format(*out_params)


    def __repr__(self):
        return self.__str__()


    def marginals(self) -> _np.ndarray:
        """Compute the marginal distribution of the parameter process."""
        pmf_x = _stats.poisson.pmf(self._support[:, None], self.lambda_)
        marginal_distrs = self.delta_ * pmf_x
        return marginal_distrs


    def get_inti_params(self):
        """Return a tuple of HMM parameters."""
        return (self._init_lambda, self._init_gamma, self._delta)


    def natural_params(self, w_params):
        """Transform working params of PoissonHMM to natural params.

            Params:
                w_params    (np.ndarray) of working parameters.

            Return:
                (lambda_, gamma_, delta_) tuple of poisson means,
                transition probability matrix and stationary distribution.
        """
        bt = _np.exp(w_params)
        bt_lambda = bt[:self.m]
        bt_gamma = _np.eye(self.m)
        _tools.set_offdiag(bt_gamma, bt[self.m:])
        bt_gamma /= bt_gamma.sum(axis=1, keepdims=True)
        return bt_lambda, bt_gamma, _utils.calculate_delta(bt_gamma)


    def working_params(self, lambda_, gamma_):
        '''Transform PoissonHMM natural parameters to working params for
            unconstrained optimization.

            The means of the state dependend distributions and are constrained
            to lambda_i > 0. The entries of the transition probability matrix
            are constrained by
                1. sum(gamma_i) = 1.
                2. gamma_ij is element of [0, 1.]
            This function removes these constraints using a generalized logit
            transform. The resulting array of working parameters has size equal
            to m*(m-1).

            Params:
                lambda_    (np.ndarray) Means of state dependend distributions.
                gamma_     (np.ndarray) Transition probability matrix.

            Return:
                (np.ndarray) of working params.
        '''
        w_lambda = _np.log(lambda_)
        w_gamma = _utils.transform_gamma(gamma_)
        return _np.concatenate((w_lambda, w_gamma))


    def _log_likelihood(self, working_params):
        '''Calculates the log-likelihood of a model given
        a set of realisations m.'''
        if self.m == 1:
            # with m=1 exp(parvect)=1
            return -_np.sum(_np.log(_stats.poisson.pmf(self.x, 1)))
        else:
            _lambda, _gamma, _delta = self.natural_params(working_params)

            # Probabilities of each realization given each mean under the
            # poisson distribtion
            poisson_probs = _np.array([_stats.poisson.pmf(self.x, mean)
                                      for mean in _lambda]).T

            # check for NaN's and occasionally replace them by 1
            poisson_probs[_np.isnan(poisson_probs)] = 1
            poisson_probs[_np.where(poisson_probs <= 1e-30)] = 1e-30

            # init
            l_scaled = 0
            psi = _delta

            # Calculate the likelihood
            # See Zucchini(2009), p. 46 for explanation
            for re in poisson_probs:
                psi = _np.multiply(psi @ _gamma, re)
                _dsum = _np.sum(psi)
                if _dsum == 0.:
                    _dsum = 1
                l_scaled += _np.log(_dsum)    # avoid underflow
                psi /= _dsum

        if _np.isnan(l_scaled):
            raise ValueError('Bad likelihood.')
        else:
            return -l_scaled


    def fit(self, method='direct'):
        '''Estimate parameters of a PoissonHMM by direct minimization of
           the negative log likelihood.'''

        # Optimizer options
        # bounds = tuple([(0, None) for i in self._working_params])
        opt = {'disp': self.verbose, 'maxiter': 100}

        try:
            ml = minimize(self._log_likelihood,
                          x0=self.working_params(self._init_lambda,
                                                 self._init_gamma),
                          method='Powell', options=opt)
        except ValueError:
            raise _except.ModelFailedError

        # transform back the parameters
        ml.x = self.natural_params(ml.x)

        # get length and sum of the parameter vector in order to calculate AIC
        # and BIC
        n_param = len(ml.x)
        sum_param = _np.nansum(self.x)

        lambda_sorted = _np.sort(ml.x[0])
        if not _np.all(lambda_sorted == ml.x[0]):

            self.lambda_ = lambda_sorted
            self.gamma_ = _utils.sort_param(ml.x[0], ml.x[1])
            self.delta_ = _utils.sort_param(ml.x[0], ml.x[2])
        else:
            self.lambda_ = ml.x[0]
            self.gamma_ = ml.x[1]
            self.delta_ = ml.x[2]

        self.nit = ml.nit
        self.llk = ml.fun
        self.status = ml.status
        self.success = ml.success
        self.aic = 2 * (self.llk + n_param)
        self.bic = 2 * self.llk + n_param * _np.log(sum_param)

        self.global_decoding = viterbi(self, self.x)
        self.trained = True
        self.theta = (self.lambda_, self.gamma_, self.delta_)

        return self.success

    def fit_EM(self):
        theta = (self._init_lambda, self._init_gamma, self._init_delta)
        self.lambda_, self.gamma_, self.delta_, self.llk, self.success = _poisson_core.poisson_EM(self.x, self.m, theta)
        if self.success:
            self.delta_ = _utils.calculate_delta(self.gamma_)
            self.theta = (self.lambda_, self.gamma_, self.delta_)

        n_params = self.m * self.m + 2 * self.m
        sum_param = _np.nansum(self.x)

        self.aic = -2 * self.llk + 2 * n_params
        self.bic = -2 * self.llk + n_params * _np.log(self.x.size)
        self.global_decoding = _poisson_core.poisson_viterbi(self, self.x)
        self.trained = True

    def nice(self):
        print(self.__str__())


    def plot(self, bins=25):
        '''Plot the marginal distributions of the PoissonHMM.
            Params:
                bins    (int) Number of bins in histogram.
            Return:
                (fig, ax)    Plot context.'''
        ax = aplot.marginal_distr(self.x, self.lambda_,
                                  self.delta_, bins=bins)
        return ax


    def sample(self, n: int, iota=None) -> _np.ndarray:
        '''Draw `n` samples form HMM.

        Params:
            n     (int)     Number of samples to draw.
            iota  (ndarray) Initial distribution. If `iota` is None
                            the model is assumed to be stationary, hence
                            the stationary distribution delta is set as iota.

        Return:
            (ndarray) of `n` samples.
        '''

        if iota is None:
            iota = self.delta_

        x = _np.zeros(n, dtype=int)

        states = range(self.m)
        # setup a discrete iota-distributed random variable
        rvInit = _stats.rv_discrete(values=(states, iota))

        # setup a rv for each state, i. e. the parameter process
        param_process = {}
        for state, row in zip(states, self.gamma_):
            param_process[state] = _stats.rv_discrete(values=(states, row))

        # setup a poisson distributed rv for each mean in lambda, i. e.
        # the state-dependend process
        sd_process = {}
        for state, lam in zip(states, self.lambda_):
            sd_process[state] = _stats.poisson(lam)

        # draw first sample given initial distribution
        state = rvInit.rvs(1)
        x[0] = sd_process[state].rvs(1)

        for i in range(1, n):
            state = param_process[state].rvs(1)
            x[i] = sd_process[state].rvs(1)

        return x


def hmm_distance(theta1: tuple, theta2: tuple, n:int=200) -> float:
    x1 = sample(theta1, n)
    x2 = sample(theta2, n)

    d12 =  (LL(x2, theta1) - LL(x2, theta2)) / len(x2)
    d21 =  (LL(x1, theta2) - LL(x1, theta1)) / len(x1)

    return -(d12 + d21)/2


def LL(x, theta):
    """Compute the log likelihood of the the model parameters.

    This function is only suitable for models with more then one
    state (m > 1).

    Params:
        x        (array) of observations
        theta    (tuple) Parameters (lambda, gamma, delta)

    Return:
        (float) log likelihood.
    """
    # initial Log-likelihood
    L = []

    lambda_, gamma_, delta_ = theta

    # state dependend probabilities given the observations
    Pr_x = _stats.poisson.pmf(x, lambda_[:, None]).T
    Pr_x[~_np.isfinite(Pr_x)] = 1.

    # n-step distribution for n=0 the initial distribution
    n_distr = delta_.copy()

    for Pr_xm in Pr_x:

        # n-step probabilities times the probs. of observing x in state m
        Pr_n = (n_distr @ gamma_) * Pr_xm

        # Probability of observing x_n in step n
        # If the sum over all states is 0 set it to 1.
        # This does not increase the result since we're calculating the log likelihood
        foo = Pr_n.sum()
        sum_Pr_n = foo if foo > 0 else 1.

        # update the likelihood and take the logrithm later on the whole array
        L.append(sum_Pr_n)

        # Pr_n is not a probability distribution, thus
        n_distr = Pr_n / sum_Pr_n

    return _np.log(L).sum()


def sample(theta: tuple, n: int, iota=None) -> _np.ndarray:
    '''Draw `n` samples form PoissonHMM.

    Params:
        mod   (tuple)      (lambda, gamma, delta)
        n     (int)        Number of samples to draw.
        iota  (ndarray)    Initial distribution. If `iota` is None
                           the model is assumed to be stationary, hence
                           the stationary distribution delta is set as iota.
    Return:
        (ndarray) of `n` samples.
    '''

    _lambda, _gamma, _delta = theta
    _m = len(_lambda)

    if iota is None:
        iota = _delta

    x = _np.zeros(n, dtype=int)

    states = range(_m)

    # setup a discrete iota-distributed random variable
    # and determine first state
    state = _stats.rv_discrete(values=(states, iota)).rvs()

    # setup parameter and state dependend process
    param_process = _np.empty(_m, dtype='object')
    sd_process = _np.empty(_m, dtype='object')

    params = zip(_gamma, _lambda)
    for i, (gi, mi) in enumerate(params):
        param_process[i] =  _stats.rv_discrete(values=(states, gi)).freeze()
        sd_process[i] = _stats.poisson.freeze(mi)

    # draw samples
    for i in range(0, n):
        x[i] = sd_process[state].rvs()
        state = param_process[state].rvs()

    return x


