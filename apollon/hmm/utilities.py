#!/usr/bin/env python3

"""utilities.py

(c) Michael Blaß 2016

Utility functions for HMMs

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

    Deprecated:
        is_tpm              Test whether matrix is row-stoachastic.
        sort_param
        transform_gamma     Transform tpm to working params.
        backt_trans_gamma       Back transform working params to tpm.
"""


import numpy as _np
from scipy import linalg as _linalg
from scipy import stats as _stats
from typing import Iterable

from apollon import tools as _tools



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

"""
----------------- Deprecated API ---------------------------------
"""

def is_tpm(mat):
    '''Test whether `mat` is a transition probability matrix.

        Tpms in first order markov chains must be two-dimensional,
        quadratic matrices with each row summing up to exactly 1.

        Params:
            mat    (np.ndarray) Matrix to test

        Return:
            True if `mat` is tpm else eaise LinAlgError.'''
    if mat.ndim == 2:
        x, y = mat.shape
        if x * y == x * x:
            if _np.isclose(mat.sum(axis=1).sum(), x):
                return True
            else:
                raise _linalg.LinAlgError('Matrix is not row-stoachastic.')
        else:
            raise _linalg.LinAlgError('Matrix is not quadratic.')
    else:
        raise _linalg.LinAlgError('Matrix must be two-dimensional.')




def sort_param(m_key, m_param):
    '''Sort one- or two-dimensional parameter array according to a unsorted
        1-d array of distribution parameters.

        In some cases the estimated distribution parameters are not in order.
        The transition probability matrix and the distribution parameters have
        then to be reorganized according to the array of sorted values.

        Params:
            m_key      (np.ndarray) Messed up array of parameters.
            m_parma    (np.ndarray) Messed up param to sort.

        Return:
            (np.ndarray) Reordered parameter.'''
    _param = _np.empty_like(m_param)

    # sort 1d
    if _param.ndim == 1:
        for i, ix in enumerate(_np.argsort(m_key)):
            _param[i] = m_param[ix]
        return _param

    # sort 2-d
    elif _param.ndim == 2:
        for i, ix in enumerate(_np.argsort(m_key)):
            for j, jx in enumerate(_np.argsort(m_key)):
                _param[i, j] = m_param[ix, jx]
        return _param
    else:
        raise ValueError('m_param must be one or two dimensinal.')


def transform_gamma(_gamma):
    '''Transform tpm to working parameters for unconstrained optimization.

        Params:
            _gamma    (np.ndarray) transition probability matrix.

        Return:
            (np.nadarray)    _gamma.shape[0]**2-_gamma.shape working params.'''
    foo = _np.log(_gamma / _gamma.diagonal()[:, None])
    w_gamma = _tools.get_offdiag(foo)
    return w_gamma

def back_trans_gamma(w_gamma, m):
    '''Back transform working params to a tpm.

        Params:
            w_gamma    (np.ndarray) of working params.
            m          (int) Number of states.

        Return:
            (np.ndarray)    Transition probability matrix.'''
    vals = _np.exp(w_gamma)
    bt_gamma = _np.eye(m)
    _tools.set_offdiag(bt_gamma, vals)
    bt_gamma /= bt_gamma.sum(axis=1, keepdims=True)
    return bt_gamma
