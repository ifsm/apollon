import os as _os
import pkg_resources as _pkg


__version__ = _pkg.get_distribution('apollon').version

APOLLON_PATH = _os.path.dirname(_os.path.realpath(__file__))
