"""
Pickle IO
"""

import pathlib
import pickle
from typing import Any

from .. typing import PathType


def load_pickle(path: PathType) -> Any:
    """Load a pickled file

    Warning:
        Only unpickle files from trusted sources. Unpickling can execute
        arbitrary code.

    Args:
        path:  Path to file.

    Returns:
        Unpickled object
    """
    path = pathlib.Path(path)
    with path.open('rb') as file:
        data = pickle.load(file)
    return data


def dump_pickle(data: Any, path: PathType) -> None:
    """Pickles data to path.

    Args:
        data:  Pickleable object.
        path:  Path to save the file.
    """
    path = pathlib.Path(path)
    with path.open('wb') as file:
        pickle.dump(data, file)
