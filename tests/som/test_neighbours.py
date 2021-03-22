import unittest

from hypothesis import strategies as hst
import numpy as np
from numpy.spatial import distance
import scipy as sp

from apollon.som import utilities as asu
from apollon.som.som import IncrementalMap


class TestIsNeighbour(unittest.TestCase):
    def setUp(self):
        self.som = IncrementralMap((10, 10, 3), 100, 0.5, 5)



