"""
datasets.py -- Load test data sets.
Copyright (C) 2018  Michael Blaß <michael.blass@uni-hamburg.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
