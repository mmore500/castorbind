"""Python bindings for the C++ core of the castor R package.

castor's functions are in ``castorbind.legacy``. castorbind is versioned
independently of castor; ``castor_version`` gives the version of castor it
wraps.
"""

from . import legacy
from ._core import _castor_version as castor_version

__author__ = "Matthew Andres Moreno"
__email__ = "m.more500@gmail.com"
__version__ = "0.2.1"

__all__ = ["castor_version", "legacy"]
