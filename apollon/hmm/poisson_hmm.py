#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""apollon/hmm/poisson_hmm.py

(c) Michael Blaß 2016


A simple implementation of a PoissonHMM including all helper functions.

Classes:
    PoissonHmm          Fitting PoissonHmm to given data.
"""


from collections import deque
import sys

import numpy as _np
from numpy import linalg as _linalg
from scipy import stats
from scipy.optimize import minimize, fmin_powell, approx_fprime

from apollon.hmm.hmm_base import HMM_Base as _HMM_Base
from apollon.hmm import utilities as _utils
# from apollon.hmm import viterbi
from apollon import aplot
from apollon import apollon_globals
from apollon import exceptions as _expect
from apollon import tools as _tools


class PoissonHmm(_HMM_Base):

    __slots__ = ['x', 'm', 'params',
                 '_init_lambda', '_init_gamma', '_init_delta',
                 'lambda_', 'gamma_', 'delta_',
                 'mllk', 'aic', 'bic',
                 'distr_params', 'model_params']

    def __init__(self, x, m, init_lambda=None, init_gamma=None,
                 init_delta=None, guess='quantile', verbose=True):

        super().__init__(x, m, init_gamma, init_delta,
                         guess='quantile', verbose=verbose)

        self.distr_params = ['lambda_']
        self.model_params = ['m', 'gamma_', 'delta_']

        # Poisson distribution is for integer data only
        if x.dtype.type is _np.int_:
            self.x = x
        else:
            if self.verbose:
                print('Warning! PoissonHMM is defined for integer time series',
                      ', only. Input has been cast to int.')
            self.x = _np.round(x).astype(int)

        # Initialize distriution parameter
        if guess in apollon_globals._lambda_guess_methods:
            if guess == 'linear':
                self._init_lambda = _utils.guess_linear(self.x, m) \
                    if init_lambda is None else init_lambda
            elif guess == 'quantile':
                self._init_lambda = _utils.guess_qunatile(self.x, m) \
                    if init_lambda is None else init_lambda
        else:
            raise ValueError('Method <{}> not supported.'.format(guess))

    def __str__(self):

        if self.trained:
            out_str = ('Lambda\n{}\n\nDelta\n{}\n\nGamma\n{}\n\n' +
                       '{:15}{:15}{:15}\n{:<15}{:<15}{:<15}')

            out_params = (self.lambda_.round(2), self.delta_.round(2),
                          self.gamma_.round(2), 'Mllk', 'AIC', 'BIC',
                          self.mllk.round(3), self.aic.round(3),
                          self.bic.round(3))
        else:
            out_str = 'init_Lambda\n{}\n\ninit_Delta\n{}\n\ninit_Gamma\n{}'
            out_params = (self._init_lambda.round(2),
                          self._init_delta.round(2),
                          self._init_gamma.round(2))

        return out_str.format(*out_params)

    def __repr__(self):
        return self.__str__()

    def get_params(self):
        '''Return a tuple of HMM parameters.'''
        return (self._init_lambda, self._init_gamma, self._delta)


    def natural_params(self, w_params):
        '''Transform working params of PoissonHMM to natural params.

            Params:
                w_params    (np.ndarray) of working parameters.

            Return:
                (lambda_, gamma_, delta_) tuple of poisson means,
                transition probability matrix and stationary distribution.
        '''
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
            return -_np.sum(_np.log(stats.poisson.pmf(self.x, 1)))
        else:
            _lambda, _gamma, _delta = self.natural_params(working_params)

            # Probabilities of each realization given each mean under the
            # poisson distribtion
            poisson_probs = _np.array([stats.poisson.pmf(self.x, mean)
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

    def train_MLLK(self):
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
            raise _expect.ModelFailedError

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
        self.mllk = ml.fun
        self.status = ml.status
        self.success = ml.success
        self.aic = 2 * (self.mllk + n_param)
        self.bic = 2 * self.mllk + n_param * _np.log(sum_param)
        # self.decoding = viterbi(self, self.x)
        self.trained = True
        return self.success

    def nice(self):
        print(self.__str__())

    def plot(self, bins=25):
        '''Plot the marginal distributions of the PoissonHMM.
            Params:
                bins    (int) Number of bins in histogram.
            Return:
                (fig, ax)    Plot context.'''
        fig, ax = aplot.marginal_distr(self.x, self.lambda_,
                                       self.delta_, bins=bins)
        return fig, ax


if __name__ == '__main__':
    a = stats.poisson.rvs(10, size=50)
    b = stats.poisson.rvs(25, size=50)
    x = _np.concatenate((a, b))
    mod = PoissonHMM(x, 2)
    print(mod)
    mod.train_MLLK()
    print('\n\n')
    print(mod)
