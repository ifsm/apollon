# 1. Languages
Apollon is written in Python3 and C. 
You will need at least Python version 3.7. There is no compatibility with any
Python2 version.
Resource critical parts are written in C by means of [Python Extension Modules](https://docs.python.org/3.7/extending/extending.html).

# 2. Contributing Python code
# 2.1 Documentation strings
Apollon adopts the [Google style for docstrings](https://google.github.io/styleguide/pyguide.html#comments).

# 2.1.1 Module level
Every module should have a module level documentation string, featuring
* module purpose
* Licence
* Author
* List of implemented functions and classes

# 2.1.2 Functions
Every function, method, or generator should have documentation string.

# 2.2 Static types
Apollon utilized static type checking by means of [mypy]
(http://www.mypy-lang.org/). Types should be added to every function. All types
are defined in [apollon/types.py](https://gitlab.rrz.uni-hamburg.de/bal7668/apollon/blob/master/apollon/types.py)
Always import the wohle module prefeixed with an underscore in public API files,
in oder to not pollute their namespaces. Always name types after the things the
objects they descibe. If you implemte a class for Bicycles the type should be
name `Bike`, or `Bicycle`. You must not postfix the type name with noise like
the word `Type`. Avoid names such as `BicycleType`. The import of the types
module avoids naming conflicts. The type for the `Bicycle` class should be
available as `_types.Bicycle`.