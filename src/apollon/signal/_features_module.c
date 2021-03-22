#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL comsar_NP_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include "correlogram.h"
#include "cdim.h"

/* Compute the correlogram of an audio signal
 *
 * Params:
 *      wlen        Length of window in samples
 *      delay       Window hop size
 *      n_lags
 *  Return 2d array
 */

static PyObject *
apollon_correlogram_delay (PyObject* self, PyObject* args)
{
    int      success    = 0;
    npy_intp window_len = 0;
    npy_intp max_offset = 0;
    npy_intp dims[]    = {0, 0};

    PyObject *op_signal = NULL;
    PyObject *op_delays = NULL;

    PyArrayObject *arr_signal = NULL;
    PyArrayObject *arr_delays = NULL;
    PyArrayObject *arr_corr   = NULL;

    if (!PyArg_ParseTuple (args, "OOkk", &op_signal, &op_delays, &window_len, &max_offset))
    {
        return NULL;
    }

    arr_signal = (PyArrayObject *) PyArray_ContiguousFromAny (op_signal, NPY_DOUBLE, 1, 1);
    if (arr_signal == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert signal array.\n");
        Py_RETURN_NONE;
    }

    arr_delays = (PyArrayObject *) PyArray_ContiguousFromAny (op_delays, NPY_LONG, 1, 1);
    if (arr_delays == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert delays array.\n");
        Py_RETURN_NONE;
    }

    dims[0] = PyArray_SIZE (arr_delays);
    dims[1] = max_offset;
    arr_corr = (PyArrayObject *) PyArray_NewFromDescr (&PyArray_Type,
                                                PyArray_DescrFromType (NPY_DOUBLE),
                                                2, dims, NULL, NULL, 0, NULL);
    if (arr_corr == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not allocate correlogram.\n");
        Py_RETURN_NONE;
    }

    success = correlogram_delay (
                (double *) PyArray_DATA (arr_signal),
                (size_t *) PyArray_DATA (arr_delays),
                (size_t) window_len,
                (size_t *) dims,
                PyArray_DATA (arr_corr));

    if (success == 0)
    {
        PyErr_SetString (PyExc_ValueError, "Correlogram failed.");
        Py_RETURN_NONE;
    }

    return (PyObject *) arr_corr;
}


static PyObject *
apollon_correlogram (PyObject* self, PyObject* args)
{
    int      success    = 0;
    npy_intp window_len = 0;
    npy_intp max_delay  = 0;
    npy_intp dims[]     = {0, 0};

    PyObject      *op_signal  = NULL;
    PyArrayObject *arr_signal = NULL;
    PyArrayObject *arr_corr   = NULL;

    if (!PyArg_ParseTuple (args, "Okk", &op_signal, &window_len, &max_delay))
    {
        return NULL;
    }

    arr_signal = (PyArrayObject *) PyArray_ContiguousFromAny (op_signal, NPY_DOUBLE, 1, 1);
    if (arr_signal == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert signal array.\n");
        Py_RETURN_NONE;
    }

    dims[0] = max_delay;
    dims[1] = PyArray_SIZE (arr_signal) - window_len - max_delay;

    arr_corr = (PyArrayObject *) PyArray_NewFromDescr (
                &PyArray_Type, PyArray_DescrFromType (NPY_DOUBLE),
                2, dims, NULL, NULL, 0, NULL);

    if (arr_corr == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not allocate correlogram.\n");
        Py_RETURN_NONE;
    }

    success = correlogram ((double *) PyArray_DATA (arr_signal),
                (size_t) window_len, (size_t *) dims, PyArray_DATA (arr_corr));

    if (success == 0)
    {
        PyErr_SetString (PyExc_ValueError, "Correlogram failed..");
        Py_RETURN_NONE;
    }

    return (PyObject *) arr_corr;
}


static PyObject *
apollon_delay_embedding_dists (PyObject *self, PyObject *args)
{
    PyObject *inp = NULL;
    npy_intp  delay = 0;
    npy_intp  m_dim = 0;

    if (!PyArg_ParseTuple (args, "Okk", &inp, &delay, &m_dim))
    {
        return NULL;
    }

    PyArrayObject *arr_inp = (PyArrayObject *) PyArray_FROM_OTF (inp,
            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_inp == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert input arrays.\n");
        Py_RETURN_NONE;
    }

    npy_intp n_vectors = PyArray_SIZE (arr_inp) - ((m_dim -1) * delay);
    npy_intp n_dists = n_vectors * (n_vectors - 1) / 2;

    /*
    PyArrayObject *dists = (PyArrayObject *) PyArray_NewFromDescr (
            &PyArray_Type, PyArray_DescrFromType (NPY_DOUBLE),
            1, &n_dists, NULL, NULL, 0, NULL);
            */
    PyArrayObject *dists = (PyArrayObject *) PyArray_ZEROS(1, &n_dists, 
            NPY_DOUBLE, 0);
    if (dists == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not allocate correlogram.\n");
        Py_RETURN_NONE;
    }

    delay_embedding_dists (PyArray_DATA (arr_inp), (size_t) n_vectors,
            (size_t) delay, (size_t) m_dim, PyArray_DATA (dists));
    
    return (PyObject *) dists;
}


static PyObject *
apollon_cdim_bader (PyObject *self, PyObject *args)
{
    PyObject *op_snd = NULL;
    npy_intp  delay;
    npy_intp  m_dim;
    npy_intp  n_bins;
    npy_intp  scaling_size;

    if (!PyArg_ParseTuple (args, "Okkkk", &op_snd, &delay, &m_dim,
                &n_bins, &scaling_size))
    {
        return NULL;
    }

    PyArrayObject *arr_snd = (PyArrayObject *) PyArray_FROM_OTF (op_snd,
            NPY_INT16, NPY_ARRAY_IN_ARRAY);

    if (arr_snd == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert input arrays.\n");
        Py_RETURN_NONE;
    }

    double cdim = corr_dim_bader (PyArray_DATA (arr_snd), (size_t) delay,
            (size_t) m_dim, (size_t) n_bins, (size_t) scaling_size);

    if (cdim < 0)
    {
        PyErr_SetString (PyExc_ValueError, "cdim_bader failed.\n");
        Py_RETURN_NONE;
    }
    return PyFloat_FromDouble (cdim);
}



static PyMethodDef
Features_Methods[] = {
    {"correlogram_delay", apollon_correlogram_delay, METH_VARARGS,
        "correlogram (signal, delays, wlen, off_max)"},
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
