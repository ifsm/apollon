"""apollon/spectral/container.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net
"""
from dataclasses import dataclass
from typing import ClassVar, Optional

from .. import io
from .. container import Params
from .. types import Schema


@dataclass
class DftParams(Params):
    """Parameter set for Discrete Fourier Transform."""
    _schema: ClassVar[Schema] = io.json.load_schema('dft_params')
    fps: int
    window: str = 'hamming'
    n_fft: Optional[int] = None


@dataclass
class StftParams(Params):
    """Parameter set for spectral transforms."""
    _schema: ClassVar[Schema] = io.json.load_schema('stft_params')
    fps: int
    window: str
    n_fft: Optional[int] = None
    n_perseg: Optional[int] = None
    n_overlap: Optional[int] = None
    extend: Optional[bool] = None
    pad: Optional[bool] = None


@dataclass
class CorrDimParams(Params):
    """Parameter set for fractal correlation dimension."""
    _schema: ClassVar[Schema] = io.json.load_schema('corrdim')
    delay: int
    m_dim: int
    n_bins: int
    scaling_size: int


@dataclass
class CorrGramParams(Params):
    """Parameter set for correlogram."""
    _schema: ClassVar[Schema] = io.json.load_schema('corrgram')
    wlen: int
    n_delay: int
    total: bool = True
