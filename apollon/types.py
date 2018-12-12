"""
types.py -- Apollon type hints.
Copyright (C) 2018  Michael Blaß <michael.blass@uni-hamburg.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pathlib
import typing
import numpy as np


Array = np.ndarray    # pylint: disable=C0103

ArrayOrStr = typing.TypeVar('ArrayOrStr', str, Array)
IterOrNone = typing.TypeVar('IterOrNone', typing.Iterable, None)
FloatOrNone = typing.TypeVar('FloatOrNone', float, None)
StrOrNone = typing.TypeVar('StrOrNone', str, None)

ParserType = typing.Tuple[typing.Dict[str, typing.Any], typing.List[str]]
PathType = typing.TypeVar('PathType', str, pathlib.Path)
PathGen = typing.Generator[PathType, None, None]
