#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
tools.py

A collection of tools often used in audio signal processing.

Clases:
    PseudoPhaseSpace    A 2D pseudo-phase space

Functions:
    get_offdiag         Return off-diag elements of square array.
    normalize           Return _normalized version of input.
    offdiag             Access to off-diagonal elements of square array.
    rowdiag             kth array diagonal sorted by array's rows.
    scale               Return scaled version of input.
    set_offdiag         Set off-diag elements of square array.
    smooth_stat         Return smoothed input.
    standardize         Return standardized input.
"""


import numpy as _np
import matplotlib.pyplot as _plt


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


def normalize(arr):
    """Normalizes an n-dim numpy array to [0, 1] regarding to the first axis.
    Thus, in a 2-d array each row is _normalized with respect to the row
    maximum.

    Param
        arr   (numerical array-like) Input signal.

    Return:
        (array) _normalized input signal.
    """
    arr = _np.atleast_1d(arr)
    if arr.ndim <= 1:
        return arr / _np.absolute(arr).max()
    else:
        return _np.vstack(row / _np.absolute(row).max() for row in arr)


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


def scale(a, end):
    """Scale a between 0 and end.

    Params:
        a    (iterable) Values to scale.
        end    (real number) Maximum value.

    Return:
        (ndarray) Scale values.
    """
    new_a = _np.atleast_1d(a) / max(a) * end
    return new_a


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


def standardize(sig):
    """Return centered and scaled version of the input.

    Params
        sig    (array-like) Input signal

    Return:
        (array) standardized input signal
    """
    return (sig - _np.mean(sig)) / _np.std(sig)


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
