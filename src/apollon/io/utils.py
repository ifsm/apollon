from typing import Any, Generator
from contextlib import contextmanager

import numpy as np

@contextmanager
def array_print_opt(*args: Any, **kwargs: Any) -> Generator[None, None, None]:
    """Set print format for numpy arrays

    Thanks to unutbu:
    https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-
    numpy-array-without-scientific-notation-and-with-given-pre
    """
    std_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**std_options)
