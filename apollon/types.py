# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""
"""
import pathlib
from typing import (Any, Dict, Generator, Iterable, List, Optional, Tuple, Union)
import numpy as _np    # type: ignore

Array = _np.ndarray    # pylint: disable = C0103

ArrayOrStr = Union[Array, str]
IterOrNone = Union[Iterable, None]

ParamsType = Dict[str, Any]
ParameterSet = Optional[ParamsType]
ParserType = Tuple[ParamsType, List[str]]
PathType = Union[str, pathlib.Path]
PathGen = Generator[PathType, None, None]
