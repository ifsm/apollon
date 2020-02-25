"""apollon/spectral/container.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net
"""
from dataclasses import dataclass
from typing import Union
from .. segment import SegmentationParams


@dataclass
class LimiterParams:
    lcf: Union[float, None] = None
    ucf: Union[float, None] = None
    ldb: Union[float, None] = None
    udb: Union[float, None] = None


@dataclass
class STParams:
    """Parameter set for spectral transforms."""
    fps: int
    window: str = 'hamming'
    n_perseg: int = 512
    n_overlap: int = 256
    n_fft: Union[int, None] = None
    extend: bool = True
    pad: bool = True
