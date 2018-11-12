#!/usr/bin/env python3

"""utilities.py

(c) Michael Blaß 2016

Utility functions for HMMs

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
