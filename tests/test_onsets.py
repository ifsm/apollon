import unittest

import numpy as np
import pandas as pd

from apollon.audio import AudioFile
from apollon.onsets.detectors import (OnsetDetector, EntropyOnsetDetector,
        FluxOnsetDetector)
from apollon.onsets.models import FluxODParams, EntropyODParams


class TestOnsetDetector(unittest.TestCase):
    def test_onset_detector(self) -> None:
        with self.assertRaises(TypeError):
            OnsetDetector()


class TestEntropyOnsetDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.snd = AudioFile('audio/beat.wav')
        self.osd = EntropyOnsetDetector(self.snd.fps)

    def test_detect(self) -> None:
        self.osd.detect(self.snd.data)

    def test_properties(self) -> None:
        self.osd.detect(self.snd.data)
        self.assertIsInstance(self.osd.odf, pd.DataFrame)
        self.assertIsInstance(self.osd.onsets, pd.DataFrame)
        self.assertIsInstance(self.osd.params, EntropyODParams)


class TestFluxOnsetDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.snd = AudioFile('audio/beat.wav')
        self.osd = FluxOnsetDetector(self.snd.fps)

    def test_detect(self) -> None:
        self.osd.detect(self.snd.data)

    def test_properties(self) -> None:
        self.osd.detect(self.snd.data)
        self.assertIsInstance(self.osd.odf, pd.DataFrame)
        self.assertIsInstance(self.osd.onsets, pd.DataFrame)
        self.assertIsInstance(self.osd.params, FluxODParams)
