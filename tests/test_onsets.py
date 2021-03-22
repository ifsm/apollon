import unittest

import numpy as np
import pandas as pd

from apollon.audio import AudioFile
from apollon.onsets import (OnsetDetector, EntropyOnsetDetector,
        FluxOnsetDetector, FilterPeakPicker)


class TestOnsetDetector(unittest.TestCase):
    def setUp(self):
        self.osd = OnsetDetector()

    def test_init(self):
        pass

    def test_to_csv(self):
        pass

    def test_to_json(self):
        pass

    def test_to_pickle(self):
        pass


class TestEntropyOnsetDetector(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat.wav')
        self.osd = EntropyOnsetDetector(self.snd.fps)

    def test_detect(self):
        self.osd.detect(self.snd.data)

    def test_odf(self):
        self.osd.detect(self.snd.data)
        self.assertIsInstance(self.osd.odf, pd.DataFrame)


class TestFluxOnsetDetector(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat.wav')
        self.osd = FluxOnsetDetector(self.snd.fps)

    def test_detect(self):
        self.osd.detect(self.snd.data)

    def test_odf(self):
        self.osd.detect(self.snd.data)
        self.assertIsInstance(self.osd.odf, pd.DataFrame)


class TestPeakPicking(unittest.TestCase):
    def setUp(self):
        self.picker = FilterPeakPicker()
        self.data = np.random.randint(0, 100, 100) + np.random.rand(100)

    def test_peaks(self):
        peaks = self.picker.detect(self.data)
        self.assertIsInstance(peaks, np.ndarray)
