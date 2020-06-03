from setuptools import setup, Extension
from setuptools.config import read_configuration
import numpy as np


config = read_configuration('./setup.cfg')

ext_features = Extension('_features',
    sources = ['src/apollon/signal/cdim.c',
               'src/apollon/signal/correlogram.c',
               'src/apollon/signal/_features_module.c'],
    include_dirs = ['include', np.get_include()])


setup(ext_modules = [ext_features])

