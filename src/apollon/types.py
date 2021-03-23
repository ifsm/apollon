"""Collection of static type hints.

:license: BSD 3 Clause
:copyright: Michael Blaß
"""
import pathlib
from typing import (Any, Collection, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, Union)
import numpy as np
from matplotlib import axes


Array = np.ndarray
ArrayOrStr = Union[Array, str]
IterOrNone = Union[Iterable, None]

ParamsType = Dict[str, Any]
PathType = Union[str, pathlib.Path]
PathGen = Generator[PathType, None, None]
Schema = Dict[str, Collection[str]]

Shape = Tuple[int, int]
SomDims = Tuple[int, int, int]
Coord = Tuple[int, int]
AdIndex = Tuple[List[int], List[int]]

Axis = axes._axes.Axes
