from setuptools import setup, Extension
from setuptools.config import read_configuration


class GetNumpyInclude:
    """Postpone the numpy import.
    This enables the package to be installed with setuptools,
    and to build numpy based extension modules during installation.
    See https://stackoverflow.com/questions/54117786/
    add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


config = read_configuration('./setup.cfg')

ext_features = Extension('_features',
    sources = ['apollon/signal/cdim.c',
               'apollon/signal/correlogram.c',
               'apollon/signal/_features_module.c'],
    include_dirs = ['include', GetNumpyInclude()])


setup(ext_modules = [ext_features])
