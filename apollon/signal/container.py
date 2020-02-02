# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/spectral/container.py
"""
from dataclasses import dataclass


@dataclass
class FftParams:
    fps: int
    n_fft: int = None
    window: str = None

@dataclass
class LimiterParams:
    lcf: float = None
    ucf: float = None
    ldb: float = None
    udb: float = None

@dataclass
class SegmentParams:
    n_per_seg: int = 512
    n_overlap: int = 256

@dataclass
class SpectrumParams(LimiterParams, FftParams):
    pass

@dataclass
class SpectrogramParams(LimiterParams, SegmentParams, FftParams):
    pass

