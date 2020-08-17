#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL comsar_NP_ARRAY_API
#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
#pragma GCC optimize("tree-vectorize")
#pragma GCC optimize("unsafe-math-optimizations")
#pragma GCC optimize("unroll-loops")
#pragma GCC diagnostic warning "-Wall"
#endif
#endif
#include <Python.h>
#include <numpy/arrayobject.h>
#include "distance.h"


/* Compute the Hellinger distance for two one-dimensional arrays.
 *
 * Params:
 *      inp_a   One-dimensional array.
 *      inp_b   One-dimensional array.
 *  Returns:
 *      float
 */
static PyObject *
apollon_som_distance_hellinger (PyObject* self, PyObject* args)
{
    int       status     = 0;
    npy_intp  n_elem     = 0;
    double    dist       = 0.0;
    PyObject *op_prob_a  = NULL;
    PyObject *op_prob_b  = NULL;

    PyArrayObject *prob_a = NULL;
    PyArrayObject *prob_b = NULL;

    if (!PyArg_ParseTuple (args, "OO", &op_prob_a, &op_prob_b))
    {
        return NULL;
    }

    prob_a = (PyArrayObject *) PyArray_ContiguousFromAny (op_prob_a, NPY_DOUBLE, 1, 1);
    if (prob_a == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert first input.\n");
        Py_RETURN_NONE;
    }

    prob_b = (PyArrayObject *) PyArray_ContiguousFromAny (op_prob_b, NPY_DOUBLE, 1, 1);
    if (prob_b == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert second input.\n");
        Py_RETURN_NONE;
    }

    n_elem = PyArray_SIZE (prob_a);
    status = hellinger (
                (double *) PyArray_DATA (prob_a),
                (double *) PyArray_DATA (prob_b),
                (size_t) n_elem,
                &dist);

    if (status < 0)
    {
        PyErr_SetString (PyExc_ValueError, "Correlogram failed.");
        Py_RETURN_NONE;
    }

    return Py_BuildValue("d", dist);
}


/* Compute the Hellinger distance for stochastic matrices
 *
 * Params:
 *      stma   One-dimensional array.
 *      stmb   One-dimensional array.
 *  Returns:
 *      Numpy array of floats.
 */
static PyObject *
apollon_som_distance_hellinger_stm (PyObject* self, PyObject* args)
{
    int      status = 0;
    npy_intp len    = 0;
    npy_intp stride = 0;
    PyObject *op_stma = NULL;
    PyObject *op_stmb = NULL;
    PyArrayObject *stma  = NULL;
    PyArrayObject *stmb  = NULL;
    PyArrayObject *dists = NULL;

    if (!PyArg_ParseTuple (args, "OO", &op_stma, &op_stmb))
    {
        return NULL;
    }

    stma = (PyArrayObject *) PyArray_ContiguousFromAny (op_stma, NPY_DOUBLE, 1, 1);
    if (stma == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert first input.\n");
        Py_RETURN_NONE;
    }

    stmb = (PyArrayObject *) PyArray_ContiguousFromAny (op_stmb, NPY_DOUBLE, 1, 1);
    if (stmb == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "Could not convert second input.\n");
        Py_RETURN_NONE;
    }

    len = PyArray_SIZE (stma);
    stride = (npy_intp) sqrt ((double) len);
    dists = (PyArrayObject *) PyArray_ZEROS(1, &stride, NPY_DOUBLE, 0);
    if (dists == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not allocate output array.\n");
        Py_RETURN_NONE;
    }

    double *dists_ptr = (double *) PyArray_DATA (dists);
    double *stma_ptr  = (double *) PyArray_DATA (stma);
    double *stmb_ptr  = (double *) PyArray_DATA (stmb);

    for (npy_intp i = 0; i < stride; i++)
    {
        status = hellinger (stma_ptr, stmb_ptr, (size_t) stride, dists_ptr);
        stma_ptr+=stride;
        stmb_ptr+=stride;
        dists_ptr++;
    }

    if (status < 0)
    {
        PyErr_SetString (PyExc_ValueError, "hellinger failed.");
        Py_RETURN_NONE;
    }

    return (PyObject *) dists;
}


static PyMethodDef
Distance_Methods[] = {
    {"hellinger", apollon_som_distance_hellinger, METH_VARARGS,
        "hellinger (prob_a, prob_b)"},
    {"hellinger_stm", apollon_som_distance_hellinger_stm, METH_VARARGS,
        "hellinger (stma, stmb)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
_distance_module = {
    PyModuleDef_HEAD_INIT,
    "_distance",
    NULL,
    -1,
    Distance_Methods
};

PyMODINIT_FUNC
PyInit__distance(void) {
    import_array();
    return PyModule_Create (&_distance_module);
}
