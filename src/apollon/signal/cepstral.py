"""
Spectral transforms
====================
"""

from typing import Self

import numpy as np


from .filter import triang_filter_bank, TriangFilterSpec
from .spectral import Dft, Stft

from ..typing import FloatArray


class Mfcc:

    def __init__(self, fps: int, n_fft: int, low: float, high: float, n_filters: int) -> None:
        self.fps = fps
        self.n_fft = n_fft
        self.fspec = TriangFilterSpec(low=low, high=high, n_filters=n_filters)
        self.fb = triang_filter_bank(self.fspec.low, self.fspec.high, self.fspec.n_filters,
                                     self.fps, self.n_fft)


    @classmethod
    def from_stft(cls, stft: Stft, spec: TriangFilterSpec) -> Self:
        """Create MFCC extractor from STFT""" 
        return NotImplemented

    @classmethod
    def from_dft(cls, dft: Dft, spec: TriangFilterSpec) -> Self:
        """Create MFCC extractor from DFT"""
        return NotImplemented

    def transform(self, data: FloatArray) -> FloatArray:
        """Transform input data"""
        return NotImplemented
