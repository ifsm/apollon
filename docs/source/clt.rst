Command line tools
==================

Apollon comes with a set of command line tools for those who do not want or
do not need to work with the API. These utilities provide access to the most
common use cases, that is, extracting features, training HMMs, training SOMS.

The command line tools, however, cannot replace the API completely. Many 
thinks like setting HMM hyper parameters are not possible at the moment.

All command line tools are invoked using the master command ``apollon``. Each
use case is implemented as a subcommand.

.. function:: apollon TRACK_FILE FEATURE_PATH [-m --mstates] [-o --outpath] 
