"""
Common tool library
"""
from datetime import datetime, timezone
import math as _math
from typing import Any, Tuple, Callable

import numpy as np

from . types import FloatArray, floatarray


def pca(data: FloatArray, n_comps: int = 2) -> Tuple[FloatArray, FloatArray,
                                                     FloatArray]:
    """Compute a PCA based on ``numpy.linalg.svd``.

    Interanlly, ``data`` will be centered but not scaled.

    Args:
        data:     Data set
        n_comps:  Number of principal components

    Returns:
        ``n_comps`` largest singular values,
        ``n_comps`` largest eigen vectors,
        transformed input data.
    """
    data_centered = data - data.mean(axis=0)
    _, vals, vects = np.linalg.svd(data_centered)

    ord_idx = np.flip(vals.argsort())[:n_comps]
    vals = vals[ord_idx]
    vects = vects[ord_idx]
    return vals, vects, data_centered @ vects.T


def assert_array(arr: FloatArray, ndim: int, size: int,     # pylint: disable=R0913
                 lower_bound: float = -np.inf,
                 upper_bound: float = np.inf,
                 name: str = 'arr') -> None:
    """Raise an error if shape of `arr` does not match given arguments.

    Args:
        arr:    Array to test
        ndim:   Expected number of dimensions
        size:   Expected total number of elements
        lower_bound:    Lower bound for array elements
        upper_bound:    Upper bound for array elements

    Raises:
        ValueError
    """
    if arr.ndim != ndim:
        raise ValueError((f'Shape of {name} does not match. Expected '
                          f'{ndim}, got {arr.ndim}.\n'))

    if arr.size != size:
        raise ValueError((f'Size of {name} does not match. Expected '
                          f'{size}, got {arr.size}.\n'))

    if np.any(arr < lower_bound):
        raise ValueError((f'Elements of {name} must be >= {lower_bound}.'))

    if np.any(arr > upper_bound):
        raise ValueError((f'Elements of {name} must be <= {upper_bound}.'))


def jsonify(inp: Any) -> Any:
    """Returns a representation of ``inp`` that can be serialized to JSON.

    This method passes through Python objects of type dict, list, str, int
    float, True, False, and None. Tuples will be converted to list by the JSON
    encoder. Numpy arrays will be converted to list using ``.to_list()``.
    On all other types, the method will try to call str() and raises
    an error.

    Args:
        inp:    Input to be jsonified

    Returns:
        Jsonified  input
    """
    valid_types = (dict, list, tuple, str, int, float)
    valid_vals = (True, False, None)

    has_type = [isinstance(inp, v_type) for v_type in valid_types]
    has_value = [inp is v_vals for v_vals in valid_vals]

    if any(has_type) or any(has_value):
        return inp

    if isinstance(inp, np.ndarray):
        return inp.tolist()

    return str(inp)


def l1_norm(arr: FloatArray) -> float:
    """Compute the L_1 norm of input vector ``x``.

    This implementation is generally faster than ``np.norm(arr, ord=1)``.
    """
    return floatarray(np.abs(arr).sum(axis=0)).item()


def normalize(arr: FloatArray, mode: str = 'array') -> FloatArray:
    """Normalize an arbitrary array_like.

    Args:
        arr:    Input signal
        mode:   Normalization mode:
                    'array' -> (default) Normalize whole array
                    'rows'  -> Normalize each row separately
                    'cols'  -> Normalize each col separately
    Return:
        Normalized input
    """

    arr = np.atleast_1d(arr)

    if mode == 'array':
        return _normalize(arr)

    if mode == 'rows':
        return np.vstack(_normalize(row) for row in arr) # type: ignore

    if mode == 'cols':
        return np.hstack(_normalize(col[:, None]) for col in arr.T) # type: ignore

    raise ValueError('Unknown normalization mode')


# TODO: This normalizes in [0, 1]; for audio we need [-1, 1]
def _normalize(arr: FloatArray) -> FloatArray:
    """Normalize array."""
    arr_min = arr.min()
    arr_max = arr.max()
    return floatarray((arr - arr_min) / (arr_max - arr_min))


def assert_and_pass(func: Callable[..., Any], arg: Any) -> Any:
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


def rowdiag(arr: FloatArray, k_diag: int = 0) -> FloatArray:
    """Get or set ``k_diag`` th diagonal of square matrix.

    Get the ``k_diag`` th diagonal of a square matrix sorted by rows or
    construct a sqare matrix with the elements of v as the main diagonal of the
    second and third dimension.

    Args:
        arr:    Square array
        k_diag: Number of diagonal

    Returns:
        Flattened diagonal
    """
    return np.diag(arr, k_diag)[:, None]


def scale(arr: FloatArray, new_min: int = 0, new_max: int = 1, axis: int = -1
          ) -> FloatArray:
    """Scale ``arr`` between ``new_min`` and ``new_max``.

    Args:
        arr:        Input array
        new_min:    Lower bound
        new_max:    Upper bound

    Return:
        One-dimensional array of transformed values.
    """
    xmax = arr.max(axis=axis, keepdims=True)
    xmin = arr.min(axis=axis, keepdims=True)

    fact = (arr-xmin) / (xmax - xmin)
    out = fact * (new_max - new_min) + new_min

    return floatarray(out)


def smooth_stat(arr: FloatArray) -> FloatArray:
    """Smooth the signal based on its mean and standard deviation.

    Args:
        arr:    Input signal

    Returns:
        Smoothed input signal
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


def standardize(arr: FloatArray) -> FloatArray:
    """Retrun z-transformed values of ``arr``.

    Args:
        arr:    Input array

    Returns:
        z-transformed array
    """
    return floatarray((arr - arr.mean(axis=0)) / arr.std(axis=0))


def time_stamp(fmt: str | None = None) -> str:
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
    """Return True if ``val`` is in between ``bounds``

    Args:
        val:    Value to test

    Returns:
       ``True``, if ``val`` is within ``bounds``, else ``False``.
    """
    return bounds[0] <= val <= bounds[1]


def within_any(val: float, windows: FloatArray) -> bool:
    """Return ``True`` if ``val`` is in any of the given ``windows``.

    Args:
        val:    Value to test
        windows: Two-dimensional array of bounds

    Returns:
    """
    lower = windows[:, 0] <= val
    upper = val <= windows[:, 1]
    cond = np.logical_and(lower, upper)
    return np.any(cond).item()
