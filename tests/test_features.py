import unittest
import numpy as np

from apollon.types import Array
from apollon.signal import features
from apollon.signal.tools import sinusoid, noise


class Test_ModuleSignalFeatures(unittest.TestCase):
    def setUp(self):
        self.data = sinusoid((300, 600), [.2, .1], fps=3000, noise=None)
        self.ecr = features.cdim(self.data, delay=14, m_dim=80, n_bins=1000,
                scaling_size=10, mode='bader')

    def test_cdim_returns_array(self):
        self.assertTrue(isinstance(self.ecr, Array))

    def test_cdim_gt_zero(self):
        self.assertTrue(np.all(self.ecr > 0))


if __name__ == '__main__':
    unittest.main()
