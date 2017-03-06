#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""hmm.py

A simple implementation of a PoissonHMM including all helper functions.

Classes:
    PoissonHmm          Fitting PoissonHmm to given data.
    PoissonHmmResults   Encapsulate Hmm results.
Functions:
    displ_array_add     Add arrays of different size.
    EM                  Estimation/Maximaization algorithm.
    forward_backward    Forwar/Backward algorithm.
    plot_hmm_data       Histogram and fitted distributions.
    rowdiag             Arbitrary diagonal sorted by rows.
    offdiag             All off-diagonal elemetns of an array.
"""

from collections import deque
import sys

import numpy as _np
from numpy import linalg as _linalg
from scipy import stats
from scipy.optimize import minimize, fmin_powell, approx_fprime
import matplotlib.pyplot as plt

from .. import apollon_globals
from . import aplot
from . import tools
from . import viterbi

__author__ = "Michael Blaß"


class PoissonHMM:
    def __init__(self, x, m, init_lambda=None, init_gamma=None,
                 init_delta=None, guess='quantile', verbose=True):
        self.verbose = verbose

        x = _np.atleast_1d(x)
        if x.dtype.type is _np.int_:
            self.x = x
        else:
            if self.verbose:
                print('Warning! PoissonHMM is defined for integer time series',
                      ', only. Input has been cast to int.')
            self.x = _np.round(x).astype(int)
        # initial parameters
        self.m = m

        if guess in apollon_globals._lambda_guess_methods:
            if guess == 'linear':
                self._init_lambda = self._init_lambda_linear(self.x, m) \
                    if init_lambda is None else init_lambda
            elif guess == 'quantile':
                self._init_lambda = self._init_lambda_qunatile(self.x, m) \
                    if init_lambda is None else init_lambda
        else:
            raise ValueError('Method <{}> not supported.'.format(guess))

        self._init_gamma = self._new_gamma(m) \
            if init_gamma is None else init_gamma
        self._delta = self._calculate_delta(m, self._init_gamma) \
            if init_delta is None else init_delta
        self._working_params = self._pn2pw()

    def __str__(self):
        out = ('Lambda\n{}\n\n' +
               'Delta\n{}\n\n' +
               'Gamma\n{}\n\n' +
               '{:15}{:15}{:15}\n' +
               '{:<15}{:<15}{:<15}').format(self.lambda_.round(2),
                                            self.delta_.round(2),
                                            self.gamma_.round(2),
                                            'Mllk', 'AIC', 'BIC',
                                            self.mllk.round(3),
                                            self.aic.round(3),
                                            self.bic.round(3))
        return out

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _calculate_delta(m, _gamma):
        '''Calculate the stationary distribution of the model.'''
        return _linalg.solve((_np.eye(m) - _gamma + 1).T, [1] * m)

    @staticmethod
    def _new_gamma(m):
        coef = _np.array([m * j for j in range(1, m + 1)])
        values = deque((sum(1 / coef[i:]) for i in range(0, m)), m)
        out = _np.zeros((m, m))
        for i in range(m):
            out[i] = values
            values.rotate()
        return out

    @staticmethod
    def _init_lambda_linear(x, m):
        '''Linearily space the lambda guesses in the range of the data.'''
        return _np.linspace(min(x), max(x), m)

    @staticmethod
    def _init_lambda_qunatile(x, m):
        '''Compute the m equally spaced percentiles from data.

        params:
            data    (array-like) data from which to compute the percentiles
            m       (int) number of HMM states

        return: m equally space percentiles
        '''
        if 3 <= m <= 100:
            pc = _np.linspace(100 / (m + 1), 100, m + 1)[:-1]
            return _np.percentile(x, pc)
        elif m == 2:
            return _np.percentile(x, [25, 75])
        elif m == 1:
            return _np.median(x)
        else:
            raise ValueError('Wrong input: m={}. 1 < m <= 100.'.format(m))

    def get_params(self):
        '''Return a tuple of HMM parameters.'''
        return (self._init_lambda, self._init_gamma, self._delta)

    def _pn2pw(self):
        '''Transforms given parameters _lambda and _gamma to a vetor of
        working parameters unsing the log function in order to avoid
        underflows.
        '''
        # log of lambda
        new_lambda = _np.log(self._init_lambda)
        if new_lambda.size > 1:
            # divide gamma by its main diagonal items in order to achive zeros
            # the main diagonal. Numpy treats vectors as they are
            # dimensionalized. R automatically takes a vector as a column
            # vector to get the items on the main diagonal of gamma. We
            # have to transpose.
            # gamma/rowdiag(gamma) == (gamma.T/diag(gamma)).T
            cv = _np.diag(self._init_gamma)
            w = self._init_gamma.T / cv

            # transform parameters
            lg = _np.log(w)

            # get all off diagonal
            # transpose matrix to get items row by row
            new_gamma = tools.offdiag(lg)

            # short: new_gamma offdiag(log(G.T/diag(G)))
            return _np.array(_np.concatenate((new_lambda, new_gamma)))
        else:
            return _np.array(new_lambda)[0]

    def _pw2pn(self, parvect):
        ''' Transforms list of working parameters back to normal '''
        recon = _np.exp(parvect)

        # first m items of parvect are lambda values
        re_lambda = recon[0:self.m]

        # construct a new mXm matrix for gamma
        re_gamma = _np.eye(self.m)

        # the remaining elements belong to th offdiag of gamma
        re_gamma = tools.offdiag(re_gamma, recon[self.m:])
        re_gamma = re_gamma.T
        for row in range(self.m):
            re_gamma[row] /= _np.sum(re_gamma[row])

        delta = _linalg.solve((_np.eye(self.m) - re_gamma + 1).T, [1] * self.m)

        return re_lambda, re_gamma, delta

    def _log_likelihood(self, working_params):
        '''Calculates the log-likelihood of a model given
        a set of realisations m.'''
        if self.m == 1:
            # with m=1 exp(parvect)=1
            return -_np.sum(_np.log(stats.poisson.pmf(self.x, 1)))
        else:
            _lambda, _gamma, _delta = self._pw2pn(working_params)

            # Probabilities of each realization
            # given each mean under the poisson
            poisson_probs = _np.array([stats.poisson.pmf(self.x, mean)
                                      for mean in _lambda]).T

            # check for NaN's and occasionally replace them by 1
            poisson_probs[_np.isnan(poisson_probs)] = 1
            poisson_probs[_np.where(poisson_probs <= 1e-30)] = 1e-30
            l_scaled = 0    # holding the likelihood

            # Calculate the likelihood
            # See Zucchini(2009), p. 46 for explanation
            for re in poisson_probs:
                _delta = _np.multiply(_delta @ _gamma, re)
                _dsum = _np.sum(_delta)
                if _dsum == 0.:
                    _dsum = 1
                l_scaled += _np.log(_dsum)    # avoid underflow
                _delta /= _dsum

        if _np.isnan(l_scaled):
            raise ValueError('Bad likelihood.')
        else:
            return -l_scaled

    def train_MLLK(self):
        '''Estimate the parameters of a Poisson-HMM by direct maximization of
           the log-likelihood.
        '''        # bounds = tuple([(0, None) for i in self._working_params])
        opt = {'disp': self.verbose, 'maxiter': 100}

        try:
            ml = minimize(self._log_likelihood, x0=self._working_params,
                          method='Powell', options=opt)
        except ValueError:
            return False

        # transform back the parameters
        ml.x = self._pw2pn(ml.x)

        # get length and sum of the parameter vector in order to calculate AIC
        # and BIC
        n_param = len(ml.x)
        sum_param = _np.nansum(self.x)

        lambda_sorted = _np.sort(ml.x[0])
        if not _np.all(lambda_sorted == ml.x[0]):
            self.lambda_ = lambda_sorted
            self.gamma_ = self._sort_gamma(ml.x[0], ml.x[1])
            self.delta_ = self._sort_delta(ml.x[0], ml.x[2])
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
        self.decoding = viterbi(self, self.x)
        return self.success

    def train_EM(self, maxiter=1000, tol=1e-6):
        '''Estimate the paramters of a Poisson-HMM by means of the
           Expectation/Maximization algorithm.

           Parameters:
               phmm       (int)
               maxiter    (int) maximum number of EM iterations
                   tol    (float) break the loop if the difference
                          between to consecutive iterations is < tol
        '''
        params = EM(self.x, self, maxiter, tol)

        if params is False:
            print('No convergence within tolerance. Training abotred.')
            return False
        else:
            self.pmeans = params[0]
            self.gamma = params[1]
            self.delta = params[2]   # self._calculate_delta(self.m, params[1])
            self.nit = params[3]
            self.mllk = params[4]
            self.status = params[5]
            self.success = params[6]
            self.aic = params[7]
            self.bic = params[8]
            return True

    def nice(self):
        print(self.__str__())

    def plot(self, bins=25):
        '''Plot the marginal distribution of the HMM.
        Params:
            bins    (int) Number of bins in histogram.
        Return:
            (fig, ax)    Plot context.'''
        fig, ax = aplot.marginal_distr(self.x, self.lambda_,
                                       self.delta_, bins=bins)
        return fig, ax

    def _sort_gamma(self, m_lambda, m_gamma):
        _gamma = _np.empty_like(m_gamma)
        for i, ix in enumerate(_np.argsort(m_lambda)):
            for j, jx in enumerate(_np.argsort(m_lambda)):
                _gamma[i, j] = m_gamma[ix, jx]
        return _gamma

    def _sort_delta(self, m_lambda, m_delta):
        _delta = _np.empty_like(m_delta)
        for i, ix in enumerate(_np.argsort(m_lambda)):
            _delta[i] = m_delta[ix]
        return _delta


class ExpHMM:
    def __init__(self, x, m, init_lambda=None, init_gamma=None,
                 init_delta=None, guess='quantile', verbose=True):
        self.verbose = verbose

        x = _np.atleast_1d(x)
        self.x = x
        # initial parameters
        self.m = m

        if guess in apollon_globals._lambda_guess_methods:
            if guess == 'linear':
                self._init_lambda = self._init_lambda_linear(self.x, m) \
                    if init_lambda is None else init_lambda
            elif guess == 'quantile':
                self._init_lambda = self._init_lambda_qunatile(self.x, m) \
                    if init_lambda is None else init_lambda
        else:
            raise ValueError('Method <{}> not supported.'.format(guess))

        self._init_gamma = self._new_gamma(m) \
            if init_gamma is None else init_gamma
        self._delta = self._calculate_delta(m, self._init_gamma) \
            if init_delta is None else init_delta
        self._working_params = self._pn2pw()

    def __str__(self):
        out = ('Lambda\n{}\n\n' +
               'Delta\n{}\n\n' +
               'Gamma\n{}\n\n' +
               '{:15}{:15}{:15}\n' +
               '{:<15}{:<15}{:<15}').format(self.lambda_.round(2),
                                            self.delta_.round(2),
                                            self.gamma_.round(2),
                                            'Mllk', 'AIC', 'BIC',
                                            self.mllk.round(3),
                                            self.aic.round(3),
                                            self.bic.round(3))
        return out

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _calculate_delta(m, _gamma):
        '''Calculate the stationary distribution of the model.'''
        return _linalg.solve((_np.eye(m) - _gamma + 1).T, [1] * m)

    @staticmethod
    def _new_gamma(m):
        coef = _np.array([m * j for j in range(1, m + 1)])
        values = deque((sum(1 / coef[i:]) for i in range(0, m)), m)
        out = _np.zeros((m, m))
        for i in range(m):
            out[i] = values
            values.rotate()
        return out

    @staticmethod
    def _init_lambda_linear(x, m):
        '''Linearily space the lambda guesses in the range of the data.'''
        return _np.linspace(min(x), max(x), m)

    @staticmethod
    def _init_lambda_qunatile(x, m):
        '''Compute the m equally spaced percentiles from data.

        params:
            data    (array-like) data from which to compute the percentiles
            m       (int) number of HMM states

        return: m equally space percentiles
        '''
        if 3 <= m <= 100:
            pc = _np.linspace(100 / (m + 1), 100, m + 1)[:-1]
            return _np.percentile(x, pc)
        elif m == 2:
            return _np.percentile(x, [25, 75])
        elif m == 1:
            return _np.median(x)
        else:
            raise ValueError('Wrong input: m={}. 1 < m <= 100.'.format(m))

    def get_params(self):
        '''Return a tuple of HMM parameters.'''
        return (self._init_lambda, self._init_gamma, self._delta)

    def _pn2pw(self):
        '''Transforms given parameters _lambda and _gamma to a vetor of
        working parameters unsing the log function in order to avoid
        underflows.
        '''
        # log of lambda
        new_lambda = _np.log(self._init_lambda)
        if new_lambda.size > 1:
            # divide gamma by its main diagonal items in order to achive zeros
            # the main diagonal. Numpy treats vectors as they are
            # dimensionalized. R automatically takes a vector as a column
            # vector to get the items on the main diagonal of gamma. We
            # have to transpose.
            # gamma/rowdiag(gamma) == (gamma.T/diag(gamma)).T
            cv = _np.diag(self._init_gamma)
            w = self._init_gamma.T / cv

            # transform parameters
            lg = _np.log(w)

            # get all off diagonal
            # transpose matrix to get items row by row
            new_gamma = tools.offdiag(lg)

            # short: new_gamma offdiag(log(G.T/diag(G)))
            return _np.array(_np.concatenate((new_lambda, new_gamma)))
        else:
            return _np.array(new_lambda)[0]

    def _pw2pn(self, parvect):
        ''' Transforms list of working parameters back to normal '''
        recon = _np.exp(parvect)

        # first m items of parvect are lambda values
        re_lambda = recon[0:self.m]

        # construct a new mXm matrix for gamma
        re_gamma = _np.eye(self.m)

        # the remaining elements belong to th offdiag of gamma
        re_gamma = tools.offdiag(re_gamma, recon[self.m:])
        re_gamma = re_gamma.T
        for row in range(self.m):
            re_gamma[row] /= _np.sum(re_gamma[row])

        delta = _linalg.solve((_np.eye(self.m) - re_gamma + 1).T, [1] * self.m)

        return re_lambda, re_gamma, delta

    def _log_likelihood(self, working_params):
        '''Calculates the log-likelihood of a model given
        a set of realisations m.'''
        if self.m == 1:
            # with m=1 exp(parvect)=1
            return -_np.sum(_np.log(stats.expon.pdf(self.x, 1)))
        else:
            _lambda, _gamma, _delta = self._pw2pn(working_params)

            # Probabilities of each realization
            # given each mean under the poisson
            poisson_probs = _np.array([stats.expon.pdf(self.x, mean)
                                      for mean in _lambda]).T

            # check for NaN's and occasionally replace them by 1
            poisson_probs[_np.isnan(poisson_probs)] = 1
            poisson_probs[_np.where(poisson_probs <= 1e-30)] = 1e-30
            l_scaled = 0    # holding the likelihood

            # Calculate the likelihood
            # See Zucchini(2009), p. 46 for explanation
            for re in poisson_probs:
                _delta = _np.multiply(_delta @ _gamma, re)
                _dsum = _np.sum(_delta)
                if _dsum == 0.:
                    _dsum = 1
                l_scaled += _np.log(_dsum)    # avoid underflow
                _delta /= _dsum

        if _np.isnan(l_scaled):
            raise ValueError('Bad likelihood.')
        else:
            return -l_scaled

    def train_MLLK(self):
        '''Estimate the parameters of a Poisson-HMM by direct maximization of
           the log-likelihood.
        '''        # bounds = tuple([(0, None) for i in self._working_params])
        opt = {'disp': self.verbose, 'maxiter': 100}

        try:
            ml = minimize(self._log_likelihood, x0=self._working_params,
                          method='Powell', options=opt)
        except ValueError:
            return False

        # transform back the parameters
        ml.x = self._pw2pn(ml.x)

        # get length and sum of the parameter vector in order to calculate AIC
        # and BIC
        n_param = len(ml.x)
        sum_param = _np.nansum(self.x)

        lambda_sorted = _np.sort(ml.x[0])
        if not _np.all(lambda_sorted == ml.x[0]):
            self.lambda_ = lambda_sorted
            self.gamma_ = self._sort_gamma(ml.x[0], ml.x[1])
            self.delta_ = self._sort_delta(ml.x[0], ml.x[2])
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
        self.decoding = viterbi(self, self.x)
        return self.success

    def train_EM(self, maxiter=1000, tol=1e-6):
        '''Estimate the paramters of a Poisson-HMM by means of the
           Expectation/Maximization algorithm.

           Parameters:
               phmm       (int)
               maxiter    (int) maximum number of EM iterations
                   tol    (float) break the loop if the difference
                          between to consecutive iterations is < tol
        '''
        params = EM(self.x, self, maxiter, tol)

        if params is False:
            print('No convergence within tolerance. Training abotred.')
            return False
        else:
            self.pmeans = params[0]
            self.gamma = params[1]
            self.delta = params[2]   # self._calculate_delta(self.m, params[1])
            self.nit = params[3]
            self.mllk = params[4]
            self.status = params[5]
            self.success = params[6]
            self.aic = params[7]
            self.bic = params[8]
            return True

    def nice(self):
        print(self.__str__())

    def plot(self, bins=25):
        '''Plot the marginal distribution of the HMM.
        Params:
            bins    (int) Number of bins in histogram.
        Return:
            (fig, ax)    Plot context.'''
        fig, ax = aplot.marginal_distr(self.x, self.lambda_,
                                       self.delta_, bins=bins)
        return fig, ax

    def _sort_gamma(self, m_lambda, m_gamma):
        _gamma = _np.empty_like(m_gamma)
        for i, ix in enumerate(_np.argsort(m_lambda)):
            for j, jx in enumerate(_np.argsort(m_lambda)):
                _gamma[i, j] = m_gamma[ix, jx]
        return _gamma

    def _sort_delta(self, m_lambda, m_delta):
        _delta = _np.empty_like(m_delta)
        for i, ix in enumerate(_np.argsort(m_lambda)):
            _delta[i] = m_delta[ix]
        return _delta
class PoissonHmmResults:
    '''Encapsulate the the results of a PoissonHmm training.'''

    __slots__ = ["lambda_", "gamma_", "delta_", "niter", "mllk", "status",
                 "success", "aic", "bic", "decoding"]

    def __init__(_lambda, _gamma, _delta, niter, mllk, status, success, aic,
                 bic, decoding):
        self.lambda_ = _lambda
        self.gamma_ = _gamma
        self.delta_ = _delta
        self.niter = niter
        self.mllk = mllk
        self.status = status
        self.success = success
        self.aic = aic
        self.bic = bic
        self.decoding = decoding


def displ_arr_add(inp_arr, disps):
    new_x = len(inp_arr)
    arr_lens = [len(arr) for arr in inp_arr]
    max_idx = _np.argmax(arr_lens)
    new_y = len(inp_arr[max_idx]) + disps[max_idx]
    out = _np.zeros((new_x, new_y))
    for i, d in enumerate(disps):
        l = len(inp_arr[i])
        out[i][d:d + l] = inp_arr[i]
    return _np.sum(out, axis=0)
