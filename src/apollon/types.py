"""apollon/types.py -- Collection of static type hints.
Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net
"""
import pathlib
from typing import (Any, Collection, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, Union)
import numpy as np

Array = np.ndarray
ArrayOrStr = Union[Array, str]
IterOrNone = Union[Iterable, None]

ParamsType = Dict[str, Any]
PathType = Union[str, pathlib.Path]
PathGen = Generator[PathType, None, None]
Schema = Dict[str, Collection[str]]
