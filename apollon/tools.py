"""
tools.py

Every day tools.


Functions:
    assert_array        Raise if array does not match given params.

    L1_Norm             Compute L1_Norm.
    normalize           Return _normalized version of input.
    in2out              Create a save from an input path.
    offdiag             Access to off-diagonal elements of square array.
    rowdiag             kth array diagonal sorted by array's rows.
    scale               Scale array between bounds.
    smooth_stat         Return smoothed input.
    standardize         Scale to zero mean and unit standard deviation.
    time_stamp          Return time stamp.
    within              Test wether val is in window.
    within_any          Test wether val is in any of windows.
"""


from datetime import datetime as _datetime
import typing as _typing

import numpy as _np

from apollon import _defaults
from apollon import types as apt


def assert_array(arr: _np.ndarray, ndim: int, size: int,     # pylint: disable=R0913
                 lower_bound: float = -_np.inf,
                 upper_bound: float = _np.inf,
                 name: str = 'arr'):
    """Raise an error if shape of `arr` does not match given arguments.

    Args:
        arr    (np.ndarray)    Array to test.
        ndim   (int)           Expected number of dimensions.
        size   (int)           Expected total number of elements.
        lower_bound (float)    Lower bound for array elements.
        upper_bound (float)    Upper bound for array elements.

    Raises:
        ValueError
    """
    if arr.ndim != ndim:
        raise ValueError(('Shape of {} does not match. Expected '
                          '{}, got {}.\n').format(name, ndim, arr.ndim))

    if arr.size != size:
        raise ValueError(('Size of {} does not match. Expected '
                          '{}, got {}.\n').format(name, size, arr.size))

    if _np.any(arr < lower_bound):
        raise ValueError(('Elements of {} must '
                          'be >= {}.'.format(name, lower_bound)))

    if _np.any(arr > upper_bound):
        raise ValueError(('Elements of {} must '
                          'be <= {}.'.format(name, upper_bound)))


def jsonify(inp: _typing.Any):
    """Returns a representation of ``inp`` that can be serialized to JSON.

    This method passes through Python objects of type dict, list, str, int
    float, True, False, and None. Tuples will be converted to list by the JSON
    encoder. Numpy arrays will be converted to list using thier .to_list() method.
    On all other types, the method will try to call str() and raises
    on error.

    Args:
        inp (Any)    Input to jsonify.

    Returns:
        (dict)  jsonified  input.
    """
    valid_types = (dict, list, tuple, str, int, float)
    valid_vals = (True, False, None)

    xx = [isintance(inp, v_type) for v_type in valid_types]
    yy = [inp is v_vals for v_vals in valid_vals]

    if any(xx) or any(yy):
        return inp

    if isinstance(inp, _np.ndarray):
        return inp.to_list()

    return str(inp)


#TODO Move to better place
def L1_Norm(arr2d: _np.ndarray) -> float:
    """Compute the L_1 norm of input vector `x`.

    This implementation is generally faster than np.norm(x, ord=1).
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

    if mode == 'rows':
        return _np.vstack(_normalize(row) for row in arr)

    if mode == 'cols':
        return _np.hstack(_normalize(col[:, None]) for col in arr.T)

    raise ValueError('Unknown normalization mode')


# TODO: This normalizes in [0, 1]; for audio we need [-1, 1]
def _normalize(arr):
    """Normalize array."""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def assert_and_pass(assert_func, arg):
    """Call `assert_func` with `arg` and return `arg`. Additionally allow arg to be None."""
    if arg is not None:
        assert_func(arg)
    return arg


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


def scale(x, new_min=0, new_max=1, axis=-1):
    """Scale `x` between `new_min` and `new_max`.

    Parmas:
        x         (np.array)          Array to be scaled.
        new_min   (real numerical)    Lower bound.
        new_max   (real numerical)    Upper bound.

    Return:
        (np.ndarray)    One-dimensional array of transformed values.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmin = x.min(axis=axis, keepdims=True)

    a = (x-xmin) / (xmax - xmin)
    y = a * (new_max - new_min) + new_min

    return y


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


def time_stamp():
    """Return default time stamp."""
    return _datetime.now().strftime(_defaults.time_stamp_fmt)


def within(val: float, bounds: _typing.Tuple[float, float]) -> bool:
    """Return True if x is in window.

    Args:
        val (float)    Value to test.

    Retrns:
        (bool)    True if ``val`` is within ``bounds``.
    """
    return bounds[0] <= val <= bounds[1]


def within_any(val: float, windows: _np.ndarray) -> bool:
    """Return True if x is in any of the given windows"""
    a = windows[:, 0] <= val
    b = val <= windows[:, 1]
    c = _np.logical_and(a, b)

    return np.any(c)
