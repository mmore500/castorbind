"""R's default RNG (Mersenne-Twister, set.seed) for castor.

castor draws through R's unif_rand(), which castorbind runs on numpy's
MT19937, the same generator R uses. This module adds R's seeding and
.Random.seed layout on top.
"""

import operator
import typing

import numpy as np

from ._core import _bit_generator, _with_castor_lock

_N = 624  # Mersenne-Twister state size


def _set_state(pos: int, key: np.ndarray) -> None:
    state = {"key": key, "pos": pos}
    _with_castor_lock(
        lambda: setattr(
            _bit_generator,
            "state",
            {"bit_generator": "MT19937", "state": state},
        )
    )


def set_seed(seed: int) -> None:
    """Seed the RNG castor uses, giving the same stream as set.seed(seed)
    in R."""
    seed = operator.index(seed)  # no floats
    if not -(2**31) < seed < 2**31:
        raise ValueError(
            "seed must be a 32-bit signed integer, excluding -2**31 (R's NA)"
        )
    # as RNG_Init in R's src/main/RNG.c: scramble, then fill the state
    # with a linear congruential generator
    seed &= 0xFFFFFFFF
    for _ in range(50):
        seed = (69069 * seed + 1) & 0xFFFFFFFF
    key = np.empty(_N + 1, dtype=np.uint32)
    for i in range(_N + 1):
        seed = (69069 * seed + 1) & 0xFFFFFFFF
        key[i] = seed
    # key[0] would be the position, which R then resets
    _set_state(_N, key[1:])


def get_rng_state() -> typing.List[int]:
    """Return the RNG state as 625 ints, laid out like .Random.seed[-1] in
    R."""
    state = _with_castor_lock(lambda: _bit_generator.state["state"])
    key = np.asarray(state["key"], dtype=np.uint32).view(np.int32)
    return [int(state["pos"])] + key.tolist()


def set_rng_state(state: typing.Sequence[int]) -> None:
    """Restore a state from get_rng_state(), or from .Random.seed[-1] in
    R."""
    values = np.asarray(state)
    if values.dtype.kind not in "iu" or values.shape != (_N + 1,):
        raise ValueError(f"state must be {_N + 1} integers")
    if not 0 <= values[0] <= _N:
        raise ValueError(f"state[0], the position, must be in [0, {_N}]")
    if values.min() < -(2**31) or values.max() >= 2**32:
        raise ValueError("state entries must fit in 32 bits")
    key = (values[1:].astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    if not key.any():
        raise ValueError("invalid RNG state: all zero")
    _set_state(int(values[0]), key)
