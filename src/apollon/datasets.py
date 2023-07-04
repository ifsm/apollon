"""
Load test data sets
"""

import collections as _collections
import os.path

import numpy as _np

from apollon import APOLLON_PATH


EarthquakesData = _collections.namedtuple('EarthquakesData',
                                  ('data', 'N', 'description'))

def load_earthquakes() -> EarthquakesData:
    """Load earthquakes dataset.

    Returns:
        (namedtuple) EqData(data, N, descr)
    """
    eq_data_path = os.path.join(APOLLON_PATH,
                                'datasets/earthquakes.data')

    eq_descr_path = os.path.join(APOLLON_PATH,
                                 'datasets/earthquakes.md')

    eq_data = _np.fromfile(eq_data_path, dtype='int64', sep=',')

    with open(eq_descr_path, encoding='utf-8') as file:
        eq_descr = ''.join(row for row in file)

    return EarthquakesData(eq_data, len(eq_data), eq_descr)
