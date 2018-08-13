#! python3

from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union


path_t = Union[str, Path]
parser_t = Tuple[Dict[str, Any], List[str]] 
path_generator_t = Generator[path_t, None, None]
