import unittest

import numpy as np
import pandas as pd

from apollon.audio import AudioFile
from apollon.onsets import OnsetDetector, FluxOnsetDetector, FilterPeakPicker



class TestOnsetDetector(unittest.TestCase):
    def setUp(self):
        ond = OnsetDetector()

    def test_init(self):
        pass

    def test_odf(self):
        self.assertIsInstance(self.flux_od.odf, pd.DataFrame)

    def test_to_csv(self):
        pass

    def test_to_json(self):
        pass

    def test_to_pickle(self):
        pass


class TestFluxOnsetDetector(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat.wav')
        self.flux_od = FluxOnsetDetector(self.snd.fps)
        self.flux_od.detect(self.snd.data)


class TestPeakPicking(unittest.TestCase):
    def setUp(self):
        self.picker = FilterPeakPicker()
        self.data = np.random.randint(0, 100, 100) + np.random.rand(100)

    def test_peaks(self):
        peaks = self.picker.detect(self.data)
        self.assertIsInstance(peaks, np.ndarray)
