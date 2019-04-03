#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


double _roughness (double d_frq, double amp_1, double amp_2);


static PyObject *
ext_roughness (PyObject *self, PyObject *args)
{
    PyObject *py_spectrogram = NULL;
    PyObject *py_frqs        = NULL;

    PyArrayObject *spctrgrm = NULL;
    PyArrayObject *frqs     = NULL;
    PyArrayObject *rghnss   = NULL;
    npy_intp      *shape    = NULL;
    size_t         n_times  = 0;
    size_t         n_frqs   = 0;

    if (!PyArg_ParseTuple (args, "OO", &py_spectrogram, &py_frqs))
    {
        return NULL;
    }

    spctrgrm = (PyArrayObject *) PyArray_FROM_OTF (py_spectrogram, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    frqs     = (PyArrayObject *) PyArray_FROM_OTF (py_frqs,        NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (spctrgrm == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not convert spectrogram buffer.\n");
        Py_RETURN_NONE;
    }

    if (frqs == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not convert frquency buffer.\n");
        Py_RETURN_NONE;
    }

    shape   = PyArray_SHAPE (spctrgrm);
    n_frqs  = shape[0];
    n_times = shape[1];
    rghnss  = (PyArrayObject *) PyArray_ZEROS (1, (npy_intp *) &n_times, NPY_DOUBLE, 0);

    if (rghnss == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Could not allocate spectrogram buffer.\n");
        Py_RETURN_NONE;
    }

    double *amp_data = PyArray_DATA (spctrgrm);
    double *frq_data = PyArray_DATA (frqs);
    double *r_data   = PyArray_DATA (rghnss);

    for (size_t t = 0; t < n_times; t++)
    {
        r_data[t] = 0.0f;

        for (size_t i = 0; i < n_frqs - 1; i++)
        {
            for (size_t j = i+1; j < n_frqs; j++)
            {
                double d_frq = fabs (frq_data[i] - frq_data[j]);
                if (d_frq >= 300.0f)
                {
                    break;
                }
                // printf ("t: %zu/%zu\t i: %zu/%zu\t j: %zu/%zut a1: %zu/%zu\t a2: %zu/%zu\t\n", t, n_times, i, n_frqs-1, j, n_frqs, i*n_times+t, n_times*n_frqs, j*n_times+t, n_times*n_frqs);
                // printf ("f1: %f, f2: %f, a1: %f, a2: %f\n", frq_data[i], frq_data[j], amp_data[i*n_times+t], amp_data[j*n_times+t]);
                r_data[t] += _roughness (d_frq, amp_data[i*n_times+t], amp_data[j*n_times+t]);
            }
        }
    }

    Py_INCREF (rghnss);
    return (PyObject *) rghnss;
}


double _roughness (double d_frq, double amp_1, double amp_2)
{
    double frq_rmax = 33.0f;
    return amp_1 * amp_2 * (d_frq / (frq_rmax * exp (-1.0))) * exp (-d_frq/frq_rmax);
}


static PyMethodDef
PF_Methods[] = {
    {"roughness", ext_roughness, METH_VARARGS,
     "roughness(spctrgrm, frqs)"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
roughness_module = {
    PyModuleDef_HEAD_INIT,
    "roughness",
    NULL,
    -1,
    PF_Methods
};

PyMODINIT_FUNC
PyInit_roughness (void)
{
    import_array ();
    return PyModule_Create (&roughness_module);
}
