# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
datasets.py -- Load test data sets.
"""

import collections as _collections
import os.path

import numpy as _np

from apollon.__init__ import APOLLON_PATH


DataSet = _collections.namedtuple('EarthquakeData',
                                  ('data', 'N', 'description'))

def load_earthquakes() -> DataSet:
    """Load earthquakes dataset.

    Returns:
        (namedtuple) EqData(data, N, descr)
    """
    eq_data_path = os.path.join(APOLLON_PATH,
                                'datasets/earthquakes.data')

    eq_descr_path = os.path.join(APOLLON_PATH,
                                 'datasets/earthquakes.md')

    eq_data = _np.fromfile(eq_data_path, dtype='int64', sep=',')

    with open(eq_descr_path) as file:
        eq_descr = ''.join(row for row in file)

    return DataSet(eq_data, len(eq_data), eq_descr)
