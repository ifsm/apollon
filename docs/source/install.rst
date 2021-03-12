Installation
***************************************
apollon can be installed on GNU/Linux, macOS, and Windows. Installation process
is similar on each of these plaforms. Note, however, that apollon contains
Python C extension modules, which have to be compile locally for GNU/Linux and
Windows users. If work on any of those platforms, please make shure that there
is a C compiler set up on your machine; otherwise the installation will fail.
In the case of macOS, a precompiled wheel is provided for the latest version
only.


Install using pip
=======================================
The Python packager manager can automatically install download and install
apollon from pip. Simply run the following command from your terminal:

.. code-block:: Bash

   python3 -m pip install apollon


Install from source
=======================================
You can also install and compile apollon directly from its sources in three
steps:

* Download the apollon source code
* Open a terminal and navigate to the apollon root directory
* Install and compile with the following command

.. code-block:: Bash

   python3 -m pip install .
