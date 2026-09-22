#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL APOLLON_NP_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include "correlogram.h"
#include "cdim.h"

/* All wrappers below follow the same pattern: scalar arguments are parsed
 * as ``Py_ssize_t`` and range-checked before use; a failed conversion or
 * allocation returns ``NULL`` with numpy's own exception set; every array
 * created or converted here is released on every path.
 */

/* Compute the correlogram of a signal for a given set of delays
 *
 * Params:
 *      signal      One-dimensional input signal
 *      delays      One-dimensional array of delays in samples
 *      wlen        Length of the correlation window in samples
 *      off_max     Number of window offsets
 *  Return 2d array of shape ``(len(delays), off_max)``
 */
static PyObject *
apollon_correlogram_delay (PyObject* self, PyObject* args)
{
    Py_ssize_t wlen       = 0;
    Py_ssize_t max_offset = 0;
    npy_intp   n_sig      = 0;
    npy_intp   n_delays   = 0;
    npy_intp   limit      = 0;
    npy_intp   dims[]     = {0, 0};
    size_t     shape[]    = {0, 0};
    size_t    *delays     = NULL;
    npy_intp  *inp_delays = NULL;

    PyObject *op_signal = NULL;
    PyObject *op_delays = NULL;

    PyArrayObject *arr_signal = NULL;
    PyArrayObject *arr_delays = NULL;
    PyArrayObject *arr_corr   = NULL;

    if (!PyArg_ParseTuple (args, "OOnn", &op_signal, &op_delays, &wlen, &max_offset))
    {
        return NULL;
    }

    if (wlen < 2)
    {
        PyErr_Format (PyExc_ValueError, "wlen must be at least 2, got %zd.", wlen);
        return NULL;
    }

    if (max_offset < 1)
    {
        PyErr_Format (PyExc_ValueError, "off_max must be positive, got %zd.", max_offset);
        return NULL;
    }

    arr_signal = (PyArrayObject *) PyArray_ContiguousFromAny (op_signal, NPY_DOUBLE, 1, 1);
    if (arr_signal == NULL)
    {
        goto fail;
    }

    arr_delays = (PyArrayObject *) PyArray_ContiguousFromAny (op_delays, NPY_INTP, 1, 1);
    if (arr_delays == NULL)
    {
        goto fail;
    }

    n_sig    = PyArray_SIZE (arr_signal);
    n_delays = PyArray_SIZE (arr_delays);
    if (n_delays < 1)
    {
        PyErr_SetString (PyExc_ValueError, "delays is empty.");
        goto fail;
    }

    if (wlen > n_sig || max_offset > n_sig - wlen)
    {
        PyErr_Format (PyExc_ValueError, "wlen (%zd) plus off_max (%zd) exceeds "
                      "the signal length (%zd).", wlen, max_offset, n_sig);
        goto fail;
    }

    /* The last sample read is sig[off_max - 1 + delay + wlen - 1]. */
    limit  = n_sig - wlen - max_offset;
    delays = PyMem_New (size_t, (size_t) n_delays);
    if (delays == NULL)
    {
        PyErr_NoMemory ();
        goto fail;
    }

    inp_delays = (npy_intp *) PyArray_DATA (arr_delays);
    for (npy_intp i = 0; i < n_delays; i++)
    {
        if (inp_delays[i] < 0 || inp_delays[i] > limit)
        {
            PyErr_Format (PyExc_ValueError, "delays[%zd] = %zd is out of range. "
                          "Delays must lie in [0, %zd] for this signal length, "
                          "wlen, and off_max.", i, inp_delays[i], limit);
            goto fail;
        }
        delays[i] = (size_t) inp_delays[i];
    }

    dims[0]  = n_delays;
    dims[1]  = max_offset;
    arr_corr = (PyArrayObject *) PyArray_SimpleNew (2, dims, NPY_DOUBLE);
    if (arr_corr == NULL)
    {
        goto fail;
    }

    shape[0] = (size_t) dims[0];
    shape[1] = (size_t) dims[1];
    if (!correlogram_delay ((double *) PyArray_DATA (arr_signal), delays,
                            (size_t) wlen, shape, (double *) PyArray_DATA (arr_corr)))
    {
        PyErr_SetString (PyExc_RuntimeError, "correlogram_delay failed.");
        goto fail;
    }

    PyMem_Free (delays);
    Py_DECREF (arr_signal);
    Py_DECREF (arr_delays);
    return (PyObject *) arr_corr;

fail:
    PyMem_Free (delays);
    Py_XDECREF (arr_signal);
    Py_XDECREF (arr_delays);
    Py_XDECREF (arr_corr);
    return NULL;
}


/* Compute the correlogram of a signal
 *
 * Params:
 *      signal      One-dimensional input signal
 *      wlen        Length of the correlation window in samples
 *      delay_max   Number of delays
 *  Return 2d array of shape ``(delay_max, len(signal) - wlen - delay_max)``
 */
static PyObject *
apollon_correlogram (PyObject* self, PyObject* args)
{
    Py_ssize_t wlen      = 0;
    Py_ssize_t max_delay = 0;
    npy_intp   n_sig     = 0;
    npy_intp   dims[]    = {0, 0};
    size_t     shape[]   = {0, 0};

    PyObject      *op_signal  = NULL;
    PyArrayObject *arr_signal = NULL;
    PyArrayObject *arr_corr   = NULL;

    if (!PyArg_ParseTuple (args, "Onn", &op_signal, &wlen, &max_delay))
    {
        return NULL;
    }

    if (wlen < 2)
    {
        PyErr_Format (PyExc_ValueError, "wlen must be at least 2, got %zd.", wlen);
        return NULL;
    }

    if (max_delay < 1)
    {
        PyErr_Format (PyExc_ValueError, "delay_max must be positive, got %zd.", max_delay);
        return NULL;
    }

    arr_signal = (PyArrayObject *) PyArray_ContiguousFromAny (op_signal, NPY_DOUBLE, 1, 1);
    if (arr_signal == NULL)
    {
        goto fail;
    }

    n_sig = PyArray_SIZE (arr_signal);
    if (wlen >= n_sig || max_delay >= n_sig - wlen)
    {
        PyErr_Format (PyExc_ValueError, "wlen (%zd) plus delay_max (%zd) must be "
                      "less than the signal length (%zd).", wlen, max_delay, n_sig);
        goto fail;
    }

    dims[0]  = max_delay;
    dims[1]  = n_sig - wlen - max_delay;
    arr_corr = (PyArrayObject *) PyArray_SimpleNew (2, dims, NPY_DOUBLE);
    if (arr_corr == NULL)
    {
        goto fail;
    }

    shape[0] = (size_t) dims[0];
    shape[1] = (size_t) dims[1];
    if (!correlogram ((double *) PyArray_DATA (arr_signal), (size_t) wlen, shape,
                      (double *) PyArray_DATA (arr_corr)))
    {
        PyErr_SetString (PyExc_RuntimeError, "correlogram failed.");
        goto fail;
    }

    Py_DECREF (arr_signal);
    return (PyObject *) arr_corr;

fail:
    Py_XDECREF (arr_signal);
    Py_XDECREF (arr_corr);
    return NULL;
}


/* Compute the condensed distance matrix of a delay embedding
 *
 * Params:
 *      inp         Input signal
 *      delay       Embedding delay in samples
 *      m_dim       Embedding dimension
 *  Return 1d array of the pairwise distances
 */
static PyObject *
apollon_delay_embedding_dists (PyObject *self, PyObject *args)
{
    Py_ssize_t delay     = 0;
    Py_ssize_t m_dim     = 0;
    npy_intp   n_inp     = 0;
    npy_intp   n_vectors = 0;
    npy_intp   n_dists   = 0;

    PyObject      *op_inp  = NULL;
    PyArrayObject *arr_inp = NULL;
    PyArrayObject *dists   = NULL;

    if (!PyArg_ParseTuple (args, "Onn", &op_inp, &delay, &m_dim))
    {
        return NULL;
    }

    if (delay < 1)
    {
        PyErr_Format (PyExc_ValueError, "delay must be positive, got %zd.", delay);
        return NULL;
    }

    if (m_dim < 1)
    {
        PyErr_Format (PyExc_ValueError, "m_dim must be positive, got %zd.", m_dim);
        return NULL;
    }

    arr_inp = (PyArrayObject *) PyArray_FROM_OTF (op_inp, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_inp == NULL)
    {
        goto fail;
    }

    n_inp = PyArray_SIZE (arr_inp);
    if (n_inp < 1 || m_dim - 1 > (n_inp - 1) / delay)
    {
        PyErr_Format (PyExc_ValueError, "An embedding of dimension %zd with delay "
                      "%zd does not fit into %zd samples.", m_dim, delay, n_inp);
        goto fail;
    }

    n_vectors = n_inp - (m_dim - 1) * delay;
    n_dists   = n_vectors * (n_vectors - 1) / 2;
    dists     = (PyArrayObject *) PyArray_ZEROS (1, &n_dists, NPY_DOUBLE, 0);
    if (dists == NULL)
    {
        goto fail;
    }

    delay_embedding_dists ((double *) PyArray_DATA (arr_inp), (size_t) n_vectors,
            (size_t) delay, (size_t) m_dim, (double *) PyArray_DATA (dists));

    Py_DECREF (arr_inp);
    return (PyObject *) dists;

fail:
    Py_XDECREF (arr_inp);
    Py_XDECREF (dists);
    return NULL;
}


/* Estimate the correlation dimension Bader-style
 *
 * Params:
 *      snd             Input signal, converted to int16
 *      delay           Embedding delay in samples
 *      m_dim           Embedding dimension
 *      n_bins          Number of histogram bins
 *      scaling_size    Distance in bins between the points of the slope
 *  Return correlation dimension estimate
 */
static PyObject *
apollon_cdim_bader (PyObject *self, PyObject *args)
{
    Py_ssize_t delay        = 0;
    Py_ssize_t m_dim        = 0;
    Py_ssize_t n_bins       = 0;
    Py_ssize_t scaling_size = 0;
    npy_intp   n_snd        = 0;
    npy_intp   n_min        = CDIM_BADER_N_SAMPLES - CDIM_BADER_BOUND;
    npy_intp   max_bin      = 0;
    double     cdim         = 0.0;

    PyObject      *op_snd  = NULL;
    PyArrayObject *arr_snd = NULL;

    if (!PyArg_ParseTuple (args, "Onnnn", &op_snd, &delay, &m_dim,
                &n_bins, &scaling_size))
    {
        return NULL;
    }

    if (delay < 1 || m_dim < 1 || n_bins < 1 || scaling_size < 1)
    {
        PyErr_Format (PyExc_ValueError, "delay (%zd), m_dim (%zd), n_bins (%zd), "
                      "and scaling_size (%zd) must be positive.", delay, m_dim,
                      n_bins, scaling_size);
        return NULL;
    }

    /* The slope is taken between the fullest bin, found among the first
     * CDIM_BADER_SEARCH (n_bins), and the bin ``scaling_size`` above it. */
    max_bin = (npy_intp) CDIM_BADER_SEARCH (n_bins) - 1;
    if (max_bin < 0)
    {
        max_bin = 0;
    }

    if (scaling_size > n_bins - 1 - max_bin)
    {
        PyErr_Format (PyExc_ValueError, "scaling_size (%zd) must not exceed %zd "
                      "for n_bins = %zd.", scaling_size, n_bins - 1 - max_bin, n_bins);
        return NULL;
    }

    arr_snd = (PyArrayObject *) PyArray_FROM_OTF (op_snd, NPY_INT16, NPY_ARRAY_IN_ARRAY);
    if (arr_snd == NULL)
    {
        goto fail;
    }

    n_snd = PyArray_SIZE (arr_snd);
    if (n_snd < n_min || m_dim - 1 > (n_snd - n_min) / delay)
    {
        PyErr_Format (PyExc_ValueError, "cdim_bader needs at least %zd + "
                      "(m_dim-1)*delay samples, got %zd with m_dim = %zd and "
                      "delay = %zd.", n_min, n_snd, m_dim, delay);
        goto fail;
    }

    cdim = corr_dim_bader ((short *) PyArray_DATA (arr_snd), (size_t) delay,
            (size_t) m_dim, (size_t) n_bins, (size_t) scaling_size);
    if (cdim < 0)
    {
        PyErr_NoMemory ();
        goto fail;
    }

    Py_DECREF (arr_snd);
    return PyFloat_FromDouble (cdim);

fail:
    Py_XDECREF (arr_snd);
    return NULL;
}



static PyMethodDef
Features_Methods[] = {
    {"correlogram_delay", apollon_correlogram_delay, METH_VARARGS,
        "correlogram_delay (signal, delays, wlen, off_max)"},
    {"correlogram", apollon_correlogram, METH_VARARGS,
        "correlogram (signal, wlen, delay_max)"},
    {"emb_dists", apollon_delay_embedding_dists, METH_VARARGS,
        "emb_dists(inp, delay, m_dim)"},
    {"cdim_bader", apollon_cdim_bader, METH_VARARGS,
     "cdim_bader (snd, delay, m_dim, n_bins, scaling_size)\n"
     "Estimate the correlation dimension Bader-style."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
_features_module = {
    PyModuleDef_HEAD_INIT,
    "_features",
    NULL,
    -1,
    Features_Methods
};

PyMODINIT_FUNC
PyInit__features(void) {
    import_array();
    return PyModule_Create (&_features_module);
}
