"""
Common tool library.
Licensed under the terms of the BSD-3-Clause license.

Copyright (C) 2019 Michael Blaß
"""
from datetime import datetime, timezone
import math as _math
from typing import Any, Tuple, Callable

import numpy as np

from . import _defaults
from . types import Array as Array


def pca(data: Array, n_comps: int = 2) -> Tuple[Array, Array, Array]:
    """Compute a PCA based on ``numpy.linalg.svd``.

    Interanlly, ``data`` will be centered but not scaled.

    Args:
        data:     Data set.
        n_comps:  Number of principal components.

    Returns:
        ``n_comps`` largest singular values,
        ``n_comps`` largest eigen vectors,
        transformed input data.
    """
    data_centered = (data - data.mean(axis=0))
    _, vals, vects = np.linalg.svd(data_centered)

    ord_idx = np.flip(vals.argsort())[:n_comps]
    vals = vals[ord_idx]
    vects = vects[ord_idx]
    return vals, vects, data_centered @ vects.T


def assert_array(arr: Array, ndim: int, size: int,     # pylint: disable=R0913
                 lower_bound: float = -np.inf,
                 upper_bound: float = np.inf,
                 name: str = 'arr'):
    """Raise an error if shape of `arr` does not match given arguments.

    Args:
        arr:    Array to test.
        ndim:   Expected number of dimensions.
        size:   Expected total number of elements.
        lower_bound:    Lower bound for array elements.
        upper_bound:    Upper bound for array elements.

    Raises:
        ValueError
    """
    if arr.ndim != ndim:
        raise ValueError(('Shape of {} does not match. Expected '
                          '{}, got {}.\n').format(name, ndim, arr.ndim))

    if arr.size != size:
        raise ValueError(('Size of {} does not match. Expected '
                          '{}, got {}.\n').format(name, size, arr.size))

    if np.any(arr < lower_bound):
        raise ValueError(('Elements of {} must '
                          'be >= {}.'.format(name, lower_bound)))

    if np.any(arr > upper_bound):
        raise ValueError(('Elements of {} must '
                          'be <= {}.'.format(name, upper_bound)))


def jsonify(inp: Any):
    """Returns a representation of ``inp`` that can be serialized to JSON.

    This method passes through Python objects of type dict, list, str, int
    float, True, False, and None. Tuples will be converted to list by the JSON
    encoder. Numpy arrays will be converted to list using thier .to_list() method.
    On all other types, the method will try to call str() and raises
    on error.

    Args:
        inp:    Input to be jsonified.

    Returns:
        Jsonified  input.
    """
    valid_types = (dict, list, tuple, str, int, float)
    valid_vals = (True, False, None)

    xx = [isinstance(inp, v_type) for v_type in valid_types]
    yy = [inp is v_vals for v_vals in valid_vals]

    if any(xx) or any(yy):
        return inp

    if isinstance(inp, np.ndarray):
        return inp.to_list()

    return str(inp)


#TODO Move to better place
def L1_Norm(arr: Array) -> float:
    """Compute the L_1 norm of input vector `x`.

    This implementation is generally faster than np.norm(arr, ord=1).
    """
    return np.abs(arr).sum(axis=0)


def normalize(arr: Array, mode: str = 'array'):
    """Normalize an arbitrary array_like.

    Args:
        arr:    Input signal.
        mode:   Normalization mode:
                    'array' -> (default) Normalize whole array.
                    'rows'  -> Normalize each row separately.
                    'cols'  -> Normalize each col separately.
    Return:
        Normalized input.
    """

    arr = np.atleast_1d(arr)

    if mode == 'array':
        return _normalize(arr)

    if mode == 'rows':
        return np.vstack(_normalize(row) for row in arr)

    if mode == 'cols':
        return np.hstack(_normalize(col[:, None]) for col in arr.T)

    raise ValueError('Unknown normalization mode')


# TODO: This normalizes in [0, 1]; for audio we need [-1, 1]
def _normalize(arr: Array) -> Array:
    """Normalize array."""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def assert_and_pass(func: Callable, arg: Any):
    """Call ``func``` with ``arg`` and return ``arg``. Additionally allow arg
    to be ``None``.

    Args:
        func:   Test function.
        arg:    Function argument.

    Returns:
        Result of ``func(arg)``.
    """
    if arg is not None:
        func(arg)
    return arg


def rowdiag(arr: Array, k: int = 0) -> Array:
    """Get or set ``k`` th diagonal of square matrix.

    Get the ``k`` th diagonal of a square matrix sorted by rows or construct a
    sqare matrix with the elements of v as the main diagonal of the second and
    third dimension.

    Args:
        arr:    Square array.
        k:      Number of diagonal.

    Returns:
        Flattened diagonal.
    """
    return np.diag(arr, k)[:, None]


def scale(arr: Array, new_min: int = 0, new_max: int = 1, axis: int = -1
          ) -> Array:
    """Scale ``arr`` between ``new_min`` and ``new_max``.

    Args:
        arr:        Array to be scaled.
        new_min:    Lower bound.
        new_max:    Upper bound.

    Return:
        One-dimensional array of transformed values.
    """
    xmax = arr.max(axis=axis, keepdims=True)
    xmin = arr.min(axis=axis, keepdims=True)

    fact = (arr-xmin) / (xmax - xmin)
    out = fact * (new_max - new_min) + new_min

    return out


def smooth_stat(arr: Array) -> Array:
    """Smooth the signal based on its mean and standard deviation.

    Args:
        arr:    Input signal.

    Returns:
        smoothed input signal.
    """
    out = []
    sig_mean = arr.mean()
    sig_std = arr.std()
    for i in arr:
        if i < sig_mean - sig_std or i > sig_mean + sig_std:
            out.append(i)
        else:
            out.append(sig_mean)

    return np.array(out)


def standardize(arr: Array) -> Array:
    """Retrun z-transformed values of ``arr``.

    Args:
        arr:    Input array.

    Returns:
        z-transformed values
    """
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def time_stamp(fmt: str = None) -> str:
    """Report call time as UTC time stamp.

    If ``fmt`` is not given, this function returns time stampes
    in ISO 8601 format.

    Args:
       fmt:  Format specification.

    Returns:
        Time stamp according to ``fmt``.
    """
    tsp = datetime.now(timezone.utc)
    if fmt is None:
        return tsp.isoformat()
    return tsp.strftime(fmt)


def within(val: float, bounds: Tuple[float, float]) -> bool:
    """Return True if x is in window.

    Args:
        val:    Value to test.

    Returns:
       ``True``, if ``val`` is within ``bounds``.
    """
    return bounds[0] <= val <= bounds[1]


def within_any(val: float, windows: Array) -> bool:
    """Return True if x is in any of the given windows.

    Args:
        val:    Value to test.
        windows: Array of bounds.

    Returns:
    """
    a = windows[:, 0] <= val
    b = val <= windows[:, 1]
    c = np.logical_and(a, b)
    return np.any(c)


def fsum(arr: Array, axis: int = None, keepdims: bool = False,
         dtype: 'str' = 'float64') -> Array:
    """Return math.fsum along the specifyed axis.

    This function supports at most two-dimensional arrays.

    Args:
        arr:      Input array.
        axis:     Reduction axis.
        keepdims: If ``True``, the output will have the same dimensionality
                  as the input.
        dtype:    Numpy data type.
    Returns:
        Sums along axis.
    """
    if axis is None:
        out = np.float64(_math.fsum(arr.flatten()))
        if keepdims:
            out = np.array(out, ndmin=arr.ndim)
    elif axis == 0:
        out = np.array([_math.fsum(col) for col in arr.T], dtype=dtype)
        if keepdims:
            out = np.expand_dims(out, 0)
    elif axis == 1:
        out = np.array([_math.fsum(row) for row in arr], dtype=dtype)
        if keepdims:
            out = np.expand_dims(out, 1)
    else:
        raise ValueError(f'``Axis is {axis} but must be 0, 1, or ``None``.')
    return out
