# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
Functions:
    assert_poisson_input    Raise if array does not conform restrictions.
    assert_st_matrix        Raise if array is not a stochastic matrix.
    assert_st_vector        Raise if array is not a stochastic vector.

    init_lambda_linear      Init linearly between min and max.
    init_lambda_quantile    Init regarding data quantiles.
    init_lambda_random      Init with random samples from data range.

    init_gamma_dirichlet    Init using Dirichlet distribution.
    init_gamma_softmax      Init with softmax of random floats.
    init_gamma_uniform      Init with uniform distr over the main diagonal.

    init_delta_dirichlet    Init using Dirichlet distribution.
    init_delta_softmax      Init with softmax of random floats.
    init_delta_stationary   Init with stationary distribution.
    init_delta_uniform      Init with uniform distribution.

    stationary_distr        Compute stationary distribution of tpm.

    get_off_diag            Return off-diagonal elements of square array.
    set_off_diag            Set off-diagonal elements of square array.
    logit_gamma             Transform tpm to logit space.
    expit_gamma             Transform tpm back from logit space.
    sort_param              Sort messed up gamma.
"""


import numpy as _np
from numpy import linalg as _linalg
from scipy import stats as _stats

from apollon import tools as _tools


def assert_poisson_input(X: _np.ndarray):
    """Check wether data is a one-dimensional array of integer values.
    Otherwise raise an exception.

    Args:
        X (np.ndarray) Data set.

    Raises:
        TypeError
        ValueError
    """
    if not isinstance(X, _np.ndarray):
        raise TypeError('`X` must be of type np.ndarray.')

    if X.ndim != 1:
        raise ValueError('Dimension of input vector must be 1.')

    if X.dtype.name != 'int64':
        raise TypeError('Input vector must be array of type int64')


def assert_st_matrix(arr: _np.ndarray):
    """Raise if `arr` is not a valid two-dimensional
    stochastic matrix.

    A stochastic matrix is a (1) two-dimensional, (2) quadratic
    matrix, with (3) elements from [0.0, 1.0] and (4) rows sums
    of exactly exactly 1.0.

    Args:
        arr (np.ndarray)    Input array.

    Raises:
        ValueError
    """
    _tools.assert_array(arr, 2, arr.size, 0.0, 1.0, 'st_matrix')

    if arr.shape[0] != arr.shape[1]:
        raise ValueError('Matrix must be quadratic.')

    if not _np.all(_np.isclose(arr.sum(axis=1), 1.0)):
        raise ValueError(('Matrix is not row-stochastic. The sum of at '
                          'least one row does not equal 1.'))


def assert_st_vector(vect: _np.ndarray):
    """Raise if `vect` is not a valid one-dimensional
    stochastic vector.

    Args:
        vect (np.ndarray)    Object to test.

    Raises:
        ValueError
    """
    if vect.ndim != 1:
        raise ValueError('Vector must be one-dimensional.')

    if not _np.isclose(vect.sum(), 1.0):
        raise ValueError('Vector is not stochastic, i. e., sum(vect) != 1.')


def assert_st_val(val: float):
    """Check wheter `val` is suitable as element of stochastic matrix.

    Args:
        val (float)    Input to check.

    Raises:
        TypeError
        ValueError
    """
    if not isinstance(val, float):
        raise TypeError('`val` must be of type float.')

    if not 0.0 <= val <= 1.0:
        raise ValueError('`val` must be within [0.0, 1.0].')


class StateDependentMeansInitializer:
    """Initializer methods for state-dependent vector of means."""

    methods = ('hist', 'linear', 'quantile', 'random')

    @staticmethod
    def hist(data: _np.ndarray, m_states: int) -> _np.ndarray:
        """Initialize state-dependent means based on a histogram of ``data``.

        The histogram is calculated with ten bins. The centers of the
        ``m_states`` most frequent bins are returned as estimates of lambda.

        Args:
            data:     Input data.
            m_states: Number of states.

        Returns:
            Lambda estimates.
        """
        frqs, bin_edges = _np.histogram(data, bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return _np.sort(bin_centers[frqs.argsort()[::-1]][:m_states])


    @staticmethod
    def linear(X: _np.ndarray, m: int) -> _np.ndarray:
        """Initialize state-dependent means with `m` linearily spaced values
        from [min(data), max(data)].

            Args:
                X    (np.ndarray)   Input data.
                m    (int)          Number of states.

            Returns:
                (np.ndarray)    Initial state-dependent means of shape (m, ).
        """
        return _np.linspace(X.min(), X.max(), m)


    @staticmethod
    def quantile(X: _np.ndarray, m: int) -> _np.ndarray:
        """Initialize state-dependent means with `m` equally spaced
        percentiles from data.

        Args:
            X    (np.ndarray) Input data.
            m    (int)        Number of HMM states.

        Returns:
            (np.ndarray)    Initial state-dependent means of shape (m, ).
        """
        if 3 <= m <= 100:
            q_range = _np.linspace(100 / (m + 1), 100, m + 1)[:-1]
            return _np.percentile(X, q_range)

        if m == 2:
            return _np.percentile(X, [25, 75])

        if m == 1:
            return _np.median(X)

        raise ValueError('Wrong input: m={}. 1 < m <= 100.'.format(m))


    @staticmethod
    def random(X: _np.ndarray, m: int) -> _np.ndarray:
        """Initialize state-dependent means with random integers from
        [min(x), max(x)[.

        Args:
            X   (np.ndarray)    Data set.
            m   (int)           Number of states.

        Retruns:
            (np.ndarray)    Initial state-dependent means of shape (m, ).
        """
        return _np.random.randint(X.min(), X.max(), m).astype(float)


class TpmInitializer:
    """Initializes transition probability matrix."""

    methods = ('dirichlet', 'softmax', 'uniform')

    @staticmethod
    def dirichlet(m: int, alpha: tuple) -> _np.ndarray:
        """
        Args:
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

        distr = (_stats.dirichlet(_np.roll(alpha, i)).rvs() for i in range(m))
        return _np.vstack(distr)


    @staticmethod
    def softmax(m: int) -> _np.ndarray:
        """Initialize `init_gamma` by applying softmax to a sample
        of random floats.

        Args:
            m   (int)   Number of states.

        Returns:
            (np.ndarray)    Transition probability matrix of shape (m, m).
        """
        init_gamma = _np.random.rand(m, m)
        return _np.exp(init_gamma) / _np.exp(init_gamma).sum(axis=1, keepdims=True)


    @staticmethod
    def uniform(m: int, diag: float) -> _np.ndarray:
        """Fill the main diagonal of `init_gamma` with `diag`. Set the
           off-diagoanl elements to the proportion of the remaining
           probability mass and the remaining number of elements per row.

            Args:
               m        (int)   Number of states.
               diag     (float) Value on main diagonal in [0, 1].

            Returns:
                (np.ndarray)    Transition probability matrix of shape (m, m).
        """
        if not isinstance(diag, float):
            raise TypeError(('Wrong type for param `diag`. '
                             'Expected <float>, got {}.\n')
                            .format(type(diag)))

        init_gamma = _np.empty((m, m))
        init_gamma.fill((1-diag) / (m-1))
        _np.fill_diagonal(init_gamma, diag)

        return init_gamma


class StartDistributionInitializer:
    """Initializes the start distribution of HMM."""

    methods = ('dirichlet', 'softmax', 'stationary', 'uniform')

    @staticmethod
    def dirichlet(m: int, alpha: tuple) -> _np.ndarray:
        """Initialize the initial distribution with a Dirichlet random sample.

        Args:
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


    @staticmethod
    def softmax(m: int) -> _np.ndarray:
        """Initialize the initial distribution by applying softmax to a sample
        of random floats.

        Args:
            m   (int)   Number of states.

        Returns:
            (np.ndarray)    Stochastic vector of shape (m, ).
        """
        rnd_vals = _np.random.rand(m)
        return _np.exp(rnd_vals) / _np.exp(rnd_vals).sum()


    @staticmethod
    def stationary(gamma_: _np.ndarray) -> _np.ndarray:
        """Initialize the initial distribution with the stationary
        distribution of `init_gamma`.

        Args:
            gamma_  (np.ndarray)    Initial transition probability matrix.

        Returns:
            (np.ndarray)    Stochastic vector of shape (m, ).
        """
        return stationary_distr(gamma_)


    @staticmethod
    def uniform(m: int) -> _np.ndarray:
        """Initialize the initial distribution uniformly.
        The initial values are set to the inverse of the number of states.

        Args:
            m   (int)   Number of states.

        Returns:
            (np.ndarray)    Stochastic vector of shape (m, ).
        """
        return _np.full(m, 1/m)


def stationary_distr(tpm: _np.ndarray) -> _np.ndarray:
    """Calculate the stationary distribution of the transition probability
    matrix `tpm`.

    Args:
        tpm (np.ndarray)    Transition probability matrix.

    Returns:
        (np.ndarray)    Stationary distribution of shape (m, ).
    """
    assert_st_matrix(tpm)
    m = tpm.shape[0]
    return _linalg.solve((_np.eye(m) - tpm + 1).T, _np.ones(m))


def get_off_diag(mat: _np.ndarray) -> _np.ndarray:
    """Return the off-diagonal elements of square array.

    Args:
        mat    (np.ndarray) square array.

    Returns:
        (np.ndarray)    mat filled with values

    Raises:
        ValueError
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix is not square.')

    mask = _np.eye(mat.shape[0], dtype=bool)
    offitems = mat[~mask]

    return offitems


def set_offdiag(mat: _np.ndarray, vals: _np.ndarray):
    """Set all off-diagonal elements of square array to
    elements of `values`.

    Args:
        mat        (np.ndarray) the matrix to fill.

    Return:
        vals     (np.ndarray) values

    Raises:
        ValueError
    """
    s_x, s_y = mat.shape
    if s_x != s_y:
        raise ValueError("Matrix must be square.")

    if vals.size != s_x * s_x - s_x:
        raise ValueError('Size of `vals` does not match shape of `mat`.')

    mask = _np.eye(s_x, dtype=bool)
    mat[~mask] = vals


def logit_tpm(tpm: _np.ndarray) -> _np.ndarray:
    """Transform tpm to logit space for unconstrained optimization.

    Note: There must be no zeros on the main diagonal.

    Args:
        tpm (np.ndarray)    Transition probability matrix.

    Returns:
        (np.nadarray)    lg_tpm of shape (1, m**2-m).
    """
    assert_st_matrix(tpm)

    logits = _np.log(tpm / tpm.diagonal()[:, None])
    lg_tpm = get_off_diag(logits)

    return lg_tpm


def expit_gamma(lg_tpm: _np.ndarray, m: int) -> _np.ndarray:
    """Transform `lg_tpm` back from logit space.

    Args:
        lg_tpm  (np.ndarray) Tpm in logit space.
        m       (int)        Number of states.

    Returns:
        (np.ndarray)    Transition probability matrix.
    """
    if not isinstance(m, int) or m < 0:
        raise ValueError('Parameter `m` must be positive integer')

    if lg_tpm.size + m != m*m:
        raise ValueError('Size of `lg_tpm` does not match number of states.')

    tpm = _np.eye(m)
    set_offdiag(tpm, _np.log(lg_tpm))
    tpm /= tpm.sum(axis=1, keepdims=True)

    return tpm


def sort_param(m_key: _np.ndarray, m_param: _np.ndarray):
    """Sort one- or two-dimensional parameter array according to a unsorted
    1-d array of distribution parameters.

    In some cases the estimated distribution parameters are not in order.
    The transition probability matrix and the distribution parameters have
    then to be reorganized according to the array of sorted values.

    Args:
        m_key      (np.ndarray) Messed up array of parameters.
        m_parma    (np.ndarray) Messed up param to sort.

    Return:
        (np.ndarray) Reordered parameter.
    """
    _param = _np.empty_like(m_param)

    # sort 1d
    if _param.ndim == 1:
        for i, key in enumerate(_np.argsort(m_key)):
            _param[i] = m_param[key]
        return _param

    # sort 2-d
    if _param.ndim == 2:
        for i, mkx in enumerate(_np.argsort(m_key)):
            for j, mky in enumerate(_np.argsort(m_key)):
                _param[i, j] = m_param[mkx, mky]
        return _param

    raise ValueError('m_param must be one or two dimensinal.')
