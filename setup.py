from setuptools import setup, Extension
import numpy as np

ext_features = Extension('_features',
    sources = ['src/apollon/signal/cdim.c',
               'src/apollon/signal/correlogram.c',
               'src/apollon/signal/_features_module.c'],
    include_dirs = ['include', np.get_include()])

ext_som_dist = Extension('_distance',
        sources = ['src/apollon/som/distance.c',
                   'src/apollon/som/_distance_module.c'],
        include_dirs = ['include', np.get_include()])

setup(ext_modules = [ext_features, ext_som_dist])
