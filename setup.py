#!/usr/bin/env  python3

from setuptools import setup, Extension
from setuptools.config import read_configuration
from numpy.distutils.misc_util import get_numpy_include_dirs


config = read_configuration('./setup.cfg')

# Extension modules
psycho_features = Extension('apollon.signal.roughness', sources=['apollon/signal/roughness.c'])

setup(include_dirs =  get_numpy_include_dirs(),
      ext_modules  = [psycho_features])
