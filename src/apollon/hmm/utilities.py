"""
HMM utility functions
"""
import numpy as _np
from numpy.typing import ArrayLike
from numpy import linalg as _linalg
from scipy import stats as _stats

from apollon import tools as _tools
from apollon.typing import Array, FloatArray, floatarray, IntArray


def assert_poisson_input(data: Array) -> None:
    """Check wether data is a one-dimensional array of integer values.
    Otherwise raise an exception.

    Args:
        data: Input array

    Raises:
        TypeError
        ValueError
    """
    if not isinstance(data, _np.ndarray):
        raise TypeError('`data` must be of type np.ndarray.')

    if data.ndim != 1:
        raise ValueError('Dimension of input vector must be 1.')

    if data.dtype.name != 'int64':
        raise TypeError('Input vector must be array of type int64')


def assert_st_matrix(arr: Array) -> None:
    """Raise if `arr` is not a valid two-dimensional
    stochastic matrix.

    A stochastic matrix is a (1) two-dimensional, (2) quadratic
    matrix, with (3) elements from [0.0, 1.0] and (4) rows sums
    of exactly exactly 1.0.

    Args:
        arr: Input array

    Raises:
        ValueError
    """
    _tools.assert_array(arr, 2, arr.size, 0.0, 1.0, 'st_matrix')

    if arr.shape[0] != arr.shape[1]:
        raise ValueError('Matrix must be quadratic.')

    if not _np.all(_np.isclose(arr.sum(axis=1), 1.0)):
        raise ValueError(('Matrix is not row-stochastic. The sum of at '
                          'least one row does not equal 1.'))


def assert_st_vector(vect: Array) -> None:
    """Raise if `vect` is not a valid one-dimensional
    stochastic vector.

    Args:
        vect: Input array

    Raises:
        ValueError
    """
    if vect.ndim != 1:
        raise ValueError('Vector must be one-dimensional.')

    if not _np.isclose(vect.sum(), 1.0):
        raise ValueError('Vector is not stochastic, i. e., sum(vect) != 1.')


def assert_st_val(val: float) -> None:
    """Check wheter `val` is suitable as element of stochastic matrix.

    Args:
        val: Input value

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
    def hist(data: IntArray, m_states: int) -> FloatArray:
        """Initialize state-dependent means based on a histogram of ``data``.

        The histogram is calculated with ten bins. The centers of the
        ``m_states`` most frequent bins are returned as estimates of lambda.

        Args:
            data:     Input data
            m_states: Number of states

        Returns:
            Lambda estimates.
        """
        frqs, bin_edges = _np.histogram(data, bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return _np.sort(bin_centers[frqs.argsort()[::-1]][:m_states])


    @staticmethod
    def linear(data: IntArray, m_states: int) -> FloatArray:
        """Initialize state-dependent means with `m_states` linearily spaced values
        from [min(data), max(data)].

            Args:
                data: Input array
                m_states: Number of model states

            Returns:
                Initial state-dependent means of shape (m_states, ).
        """
        return _np.linspace(data.min(), data.max(), m_states, dtype=_np.double)


    @staticmethod
    def quantile(data: IntArray, m_states: int) -> FloatArray:
        """Initialize state-dependent means with `m_states` equally spaced
        percentiles from data.

        Args:
            data: Input array
            m_states: Number of model states

        Returns:
            Initial state-dependent means of shape (m_states, ).
        """
        if 3 <= m_states <= 100:
            q_range = _np.linspace(100 / (m_states + 1), 100, m_states + 1)[:-1]
            return _np.percentile(data, q_range)

        if m_states == 2:
            return _np.percentile(data, [25, 75])

        if m_states == 1:
            return _np.asarray(_np.atleast_1d(_np.median(data))).astype(_np.double)

        raise ValueError(f'Wrong input: {m_states=}. Expected 1 < m_states <= 100.')


    @staticmethod
    def random(data: IntArray, m_states: int) -> FloatArray:
        """Initialize state-dependent means with random integers from
        [min(x), max(x)[.

        Args:
            data: Input array
            m_states: Number of model states

        Retruns:
            Initial state-dependent means of shape (m_states, ).
        """
        return _np.random.randint(data.min(), data.max(), m_states).astype(float)


class TpmInitializer:
    """Initializes transition probability matrix."""

    methods = ('dirichlet', 'softmax', 'uniform')

    @staticmethod
    def dirichlet(m_states: int, alpha: float | ArrayLike) -> FloatArray:
        """
        Args:
            m_states: Number of model states
            alpha: Dirichlet distribution parameters. Iterable of size `m_states`.
                   Each entry controls the probability mass that is put on the
                   respective transition.
        Returns:
            Transition probability matrix of shape (m_states, m_states).
        """
        alpha = _np.atleast_1d(alpha)

        if alpha.ndim != 1:
            raise ValueError('Wrong shape of param `alpha`. Expected float '
                             f'or array-like of shape {m_states}, got '
                             f'{alpha.ndim=}.')

        if alpha.size != m_states:
            raise ValueError('Wrong size of param `alpha`. Expected {m_states} '
                             f', got {alpha.ndim=}')

        distr = [_stats.dirichlet(_np.roll(alpha, i)).rvs()
                 for i in range(m_states)]
        return _np.vstack(distr)


    @staticmethod
    def softmax(m_states: int) -> FloatArray:
        """Initialize `init_gamma` by applying softmax to a sample
        of random floats.

        Args:
            m_states: Number of model states

        Returns:
            Transition probability matrix of shape (m_states, m_states).
        """
        gamma: FloatArray = _np.random.rand(m_states, m_states)
        _np.exp(gamma, out=gamma)
        _np.divide(gamma, gamma.sum(axis=1, keepdims=True), out=gamma)
        return gamma


    @staticmethod
    def uniform(m_states: int, diag: float) -> FloatArray:
        """Fill the main diagonal of `init_gamma` with `diag`. Set the
           off-diagoanl elements to the proportion of the remaining
           probability mass and the remaining number of elements per row.

            Args:
               m_states: Number of model states
               diag: Value on main diagonal in [0, 1].

            Returns:
                Transition probability matrix of shape (m_states, m_states).
        """
        if not isinstance(diag, float):
            raise TypeError('Wrong type for param `diag`. '
                            f'Expected <float>, got {type(diag)}.')

        init_gamma = _np.empty((m_states, m_states))
        init_gamma.fill((1-diag) / (m_states-1))
        _np.fill_diagonal(init_gamma, diag)

        return init_gamma


class StartDistributionInitializer:
    """Initializes the start distribution of HMM."""

    methods = ('dirichlet', 'softmax', 'stationary', 'uniform')

    @staticmethod
    def dirichlet(m_states: int, alpha: float | ArrayLike) -> FloatArray:
        """Initialize the initial distribution with a Dirichlet random sample.

        Args:
            m_states: Number of model states
            alpha: Dirichlet distribution params

        Returns:
            Stochastic vector of shape (`m_states`, ).
        """
        alpha = _np.atleast_1d(alpha)

        if alpha.ndim != 1:
            raise ValueError('Wrong shape of param `alpha`. Expected float or '
                             f'array-like of shape {m_states}, got '
                             f'{alpha.ndim=}')

        if alpha.size != m_states:
            raise ValueError('Wrong size of param `alpha`. Expected {m_states} '
                             f', got {alpha.size=}')

        return floatarray(_stats.dirichlet(alpha).rvs())


    @staticmethod
    def softmax(m_states: int) -> FloatArray:
        """Initialize the initial distribution by applying softmax to a sample
        of random floats.

        Args:
            m_statesa: Number of model states

        Returns:
            Stochastic vector of shape (`m_states`, )
        """
        rnd_vals: FloatArray = _np.random.rand(m_states)
        _np.exp(rnd_vals, out=rnd_vals)
        _np.divide(rnd_vals, rnd_vals.sum(), out=rnd_vals)
        return rnd_vals


    @staticmethod
    def stationary(gamma_: FloatArray) -> FloatArray:
        """Initialize the initial distribution with the stationary
        distribution of `init_gamma`.

        Args:
            gamma_: Initial transition probability matrix

        Returns:
            Stochastic vector of shape (`m_states`, )
        """
        return stationary_distr(gamma_)


    @staticmethod
    def uniform(m_states: int) -> FloatArray:
        """Initialize the initial distribution uniformly.
        The initial values are set to the inverse of the number of states.

        Args:
            m_states: Number of model states

        Returns:
            Stochastic vector of shape (`m_states`, ).
        """
        return _np.full(m_states, 1/m_states)


def stationary_distr(tpm: FloatArray) -> FloatArray:
    """Calculate the stationary distribution of the transition probability
    matrix `tpm`.

    Args:
        tpm: Transition probability matrix

    Returns:
        Stationary distribution of shape (`m_states`, ).
    """
    assert_st_matrix(tpm)
    m_states = tpm.shape[0]
    return _linalg.solve((_np.eye(m_states) - tpm + 1).T, _np.ones(m_states))


def get_off_diag(mat: FloatArray) -> FloatArray:
    """Return the off-diagonal elements of square array.

    Args:
        mat: Input array

    Returns:
        All elements of `mat` that do not lie on the main diagonal. 

    Raises:
        ValueError
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix is not square.')

    mask = _np.eye(mat.shape[0], dtype=bool)
    offitems: FloatArray = mat[~mask]

    return offitems


def set_offdiag(mat: Array, vals: Array) -> Array:
    """Set all off-diagonal elements of square array to
    elements of `values`.

    Args:
        mat: Input array

    Return:
        `mat` with all off-diagonla elements set to `val`.

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
    return mat


def logit_tpm(tpm: FloatArray) -> FloatArray:
    """Transform tpm to logit space for unconstrained optimization.

    Note: There must be no zeros on the main diagonal.

    Args:
        tpm: Transition probability matrix

    Returns:
       lg_tpm of shape (1, m_states**2-m_states).
    """
    assert_st_matrix(tpm)

    logits = _np.log(tpm / tpm.diagonal()[:, None])
    lg_tpm = get_off_diag(logits)

    return lg_tpm


def expit_gamma(lg_tpm: FloatArray, m_states: int) -> FloatArray:
    """Transform `lg_tpm` back from logit space.

    Args:
        lg_tpm: Tpm in logit space
        m_states: Number of model states

    Returns:
        Transition probability matrix
    """
    if not isinstance(m_states, int) or m_states < 0:
        raise ValueError('Parameter `m_states` must be positive integer')

    if lg_tpm.size + m_states != m_states*m_states:
        raise ValueError('Size of `lg_tpm` does not match number of states.')

    tpm = _np.eye(m_states)
    set_offdiag(tpm, _np.log(lg_tpm))
    tpm /= tpm.sum(axis=1, keepdims=True)

    return tpm


def sort_param(m_key: FloatArray, m_param: FloatArray) -> FloatArray:
    """Sort one- or two-dimensional parameter array according to a unsorted
    1-d array of distribution parameters.

    In some cases the estimated distribution parameters are not in order.
    The transition probability matrix and the distribution parameters have
    then to be reorganized according to the array of sorted values.

    Args:
        m_key: Messed up array of parameters.
        m_parma: Messed up param to sort.

    Return:
        Reordered parameter
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
