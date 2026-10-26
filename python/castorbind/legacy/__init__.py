"""castor's C++ functions, under their castor names minus the _CPP suffix.

They take castor's own flat, 0-based inputs and return dicts. R's random
number generator is included, so ``set_seed(n)`` before a castor simulator
gives the same stream as ``set.seed(n)`` in R.
"""

from .. import _core
from .._core import *  # noqa: F401
from .._rng import get_rng_state, set_rng_state, set_seed

__all__ = [name for name in dir(_core) if not name.startswith("_")] + [
    "get_rng_state",
    "set_rng_state",
    "set_seed",
]

del _core
