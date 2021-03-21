import sphinx_rtd_theme

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./src/apollon/'))


# -- Project information -----------------------------------------------------

project = 'apollon'
author = 'Michael Blaß'
copyright = '2019, Michael Blaß'

# The full version, including alpha/beta/rc tags
version = '0.1'
release = '0.1.3'

master_doc = 'index'


# -- General configuration ---------------------------------------------------
source_suffix = {'.rst': 'restructuredtext'}
language = 'en'
#nitpicky = True
numfig = True
pygments_style = 'sphinx'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
#    'sphinxcontrib.apidoc',
    'sphinx_rtd_theme']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


autodoc_default_options = {
    'inherited-members': None,
}


# -- Options for autodoc  ---------------------------------------------------
#
autodoc_member_order = 'bysource'


# -- Options for autosummary ------------------------------------------------
#



# -- Options for apidoc -----------------------------------------------------
#
#apidoc_module_dir = '../../src/apollon'
#apidoc_output_dir = 'generated/api'
#apidoc_excluded_paths = ['schema']
#apidoc_separate_modules = True
#apidoc_toc_file = False
#apidoc_module_first = True
#apidoc_extra_args = ['--no-headings']


# -- Options for Napoleon ---------------------------------------------------
#
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
