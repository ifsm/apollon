# pylint: disable = missing-class-docstring,too-few-public-methods
"""
common models
"""
from pydantic import BaseModel

from . signal.models import StftParams


class PeakPickingParams(BaseModel):
    n_before: int
    n_after: int
    alpha: float
    delta: float


class FluxOnsetDetectorParams(BaseModel):
    stft_params: StftParams
    pp_params: PeakPickingParams
