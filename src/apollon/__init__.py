"""
Apollon feature extraction framework.
"""

import os as _os
from importlib import metadata as _meta

__version__ = _meta.version("apollon")

APOLLON_PATH = _os.path.dirname(_os.path.realpath(__file__))
