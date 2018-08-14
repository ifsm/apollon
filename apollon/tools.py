#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
tools.py

Every day tools.


Functions:
    get_offdiag         Return off-diag elements of square array.
    L1_Norm             Compute L1_Norm.
    normalize           Return _normalized version of input.
    in2out              Create a save from an input path.
    offdiag             Access to off-diagonal elements of square array.
    rowdiag             kth array diagonal sorted by array's rows.
    scale               Scale array between bounds.
    set_offdiag         Set off-diag elements of square array.
    smooth_stat         Return smoothed input.
    standardize         Scale to zero mean and unit standard deviation.
    time_stamp          Return time stamp.
    within              Test wether val is in window.
    within_any          Test wether val is in any of windows.
"""


from datetime import datetime as _datetime
import numpy as _np
import matplotlib.pyplot as _plt
import pathlib as _pathlib
from typing import Iterable

from apollon import _defaults


__author__ = 'Michael Blaß'


def get_offdiag(mat):
    """Return all off-diagonal elements of square array.

    Params:
        mat    (np.ndarray) square array.

    Return:
        (np.ndarray)    mat filled with values
    """
    x, y = mat.shape
    if x != y:
        raise ValueError('Matrix is not square.')
    mask = _np.eye(x, dtype=bool)
    offitems = mat[~mask]

    return offitems


def in2out(inp_path, out_path, ext=None):
    '''Creat a save path from inp_path.'''

    if isinstance(inp_path, str):
        inp_path = _pathlib.Path(inp_path)

    if isinstance(out_path, str):
        out_path = _pathlib.Path(out_path)

    _ext = ext if ext.startswith('.') else '.' + ext
    fn = inp_path.stem if ext is None else inp_path.stem + _ext
    return out_path.joinpath(fn)


def L1_Norm(x):
    """Compute the L_1 norm of input vector `x`.

    This is implementation generally faster than np.norm(x, ord=1).
    """
    return _np.abs(x).sum(axis=0)


def normalize(arr, mode='array'):
    """Normalize an arbitrary array_like.

    Params:
        arr   (numerical array-like) Input signal.
        axis  (str) Normalization mode:
                    'array' -> (default) Normalize whole array.
                    'rows'  -> Normalize each row separately.
                    'cols'  -> Normalize each col separately.
    Return:
        (array) Normalized input.
    """

    arr = _np.atleast_1d(arr)

    if mode == 'array':
        return _normalize(arr)

    elif mode == 'rows':
        return _np.vstack(_normalize(row) for row in arr)

    elif mode == 'cols':
        return _np.hstack(_normalize(col[:, None]) for col in arr.T)

    else:
        raise ValueError('Unknown normalization mode')

# TODO: This normalizes in [0, 1]; for audio we need [-1, 1]
def _normalize(arr):
    """Normalize array."""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)



def rowdiag(v, k=0):
    """Get or set k'th diagonal of square matrix.

    Get the k'th diagonal of a square matrix sorted by rows
    or construct a sqare matrix with the elements of v as the
    main diagonal of the second and third dimension.

    Params:
        v    (array) Square matrix.
        k    (int) Number diagonal.

    Return:
        (1d-array) Values of diagonal.
    """
    return _np.diag(v, k)[:, None]


def scale(x, new_min=0, new_max=1):
    """Scale `x` between `new_min` and `new_max`.
    
    Parmas:
        x        (np.array)          One-dimensional array of values to be scales.
        newmin   (real numerical)    Lower bound.
        newmax   (real numerical)    Upper bound.
        
    Return:
        (np.ndarray)    One-dimensional array of transformed values.
    """
    xm = x.max()
    a = (new_max - new_min) / (xm - x.min())
    b = new_max - a * xm
    
    return a*x + b


def smooth_stat(sig):
    """Smooth the signal based on its mean and standard deviation.

    Params:
        sig    (array-like) Input signal.

    Return:
        (ndarray) smoothed input signal.
    """
    out = []
    sig_mean = sig.mean()
    sig_std = sig.std()
    for i in sig:
        if i < sig_mean - sig_std or i > sig_mean + sig_std:
            out.append(i)
        else:
            out.append(sig_mean)

    return _np.array(out)


def standardize(x: _np.ndarray) -> _np.ndarray:
    """Retrun z-transformed values of x.

    Params:
        x    (array) Input values

    Return:
        (array) z-transformed values
    """
    return (x - x.mean(axis=0)) / x.std(axis=0)



def set_offdiag(mat, values):
    """Set all off-diagonal of square array with elements with `values`.

    Params:
        mat        (np.ndarray) the matrix to fill.

    Return:
        values     (np.ndarray) values
    """
    values = _np.atleast_1d(values)

    x, y = mat.shape
    if x != y:
        raise ValueError("Matrix is not square.")

    mask = _np.eye(x, dtype=bool)
    offitems = mat[~mask]

    if values.size == offitems.size:
        mat[~mask] = values
    else:
        raise IndexError("Number of off-diagonal elements must equal length  \
                         of values.")

def time_stamp():
    """Return default time stamp."""
    return _datetime.now().strftime(_defaults.time_stamp_fmt)


def within(x: float, window: Iterable) -> bool:
    """Return True if x is in window."""
    return window[0] <= x <= window[1]


def within_any(x: float, windows: _np.ndarray) -> bool:
    """Return True if x is in any of the given windows"""
    a = windows[:, 0] <= x
    b = x <= windows[:, 1]
    c = _np.logical_and(a, b)

    return np.any(c)

