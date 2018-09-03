#!/usr/bin/python3
# -*- coding: utf-8 -*-

from collections import namedtuple as _namedtuple

import numpy as _np

from apollon.__init__ import APOLLON_PATH


def load_earthquakes() -> _namedtuple:
    """Load earthquakes dataset.
       Return:
            (namedtuple) EqData(data, N, descr)
    """

    # set file paths
    eq_data_path = APOLLON_PATH + '/datasets/earthquakes.data'
    eq_descr_path = APOLLON_PATH + '/datasets/earthquakes.md'

    # load data
    data = _np.fromfile(eq_data_path, dtype='uint8', sep=',')
    N = len(data)

    with open(eq_descr_path) as fobj:
        descr = ''.join(row for row in fobj)

    # construct dataset object
    EqData = _namedtuple('EarthquakeData', ('data', 'N', 'description'))
    eqd = EqData(data, len(data), descr)

    return eqd
