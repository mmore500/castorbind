"""Python bindings for the C++ core of the castor R package.

Every function castor exports from C++ is available under its castor name,
minus the _CPP suffix, taking castor's own flat, 0-based inputs and
returning dicts. R's random number generator is included, so
``set_seed(n)`` before a castor simulator gives the same stream as
``set.seed(n)`` in R.

castorbind is versioned independently of castor; ``castor_version`` gives
the version of castor it wraps.
"""

from . import _core
from ._core import *  # noqa: F401

__author__ = "Matthew Andres Moreno"
__email__ = "m.more500@gmail.com"
__version__ = "0.1.0"

__all__ = [name for name in dir(_core) if not name.startswith("_")]
