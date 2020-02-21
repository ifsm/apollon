"""apollon/spectral/container.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net
"""
from dataclasses import dataclass
from typing import Union

@dataclass
class FftParams:
    fps: int
    n_fft: Union[int, None] = None
    window: Union[str, None] = None

@dataclass
class LimiterParams:
    lcf: Union[float, None] = None
    ucf: Union[float, None] = None
    ldb: Union[float, None] = None
    udb: Union[float, None] = None

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
