Data models
============

Each onset detector has its own parameter model. These models the types and
value ranges for each parameter. They can also be used for easy serialization
of the input parameters. To this end, each model implements the methods
:code:`model_dump`, and :code:`model_dump_json`, which return the model data
as plain Python dictionary or JSON string, respectively.

.. automodule:: apollon.onsets.models
   :members:
