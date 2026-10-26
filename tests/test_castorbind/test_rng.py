import pytest

import castorbind
from castorbind import _core

from ..conftest import assert_matches_r


def test_set_seed_reproducible():
    castorbind.set_seed(7)
    first = _core._runif(5)
    castorbind.set_seed(7)
    assert _core._runif(5) == first
    castorbind.set_seed(8)
    assert _core._runif(5) != first


def test_rng_state_round_trip():
    castorbind.set_seed(1)
    state = castorbind.get_rng_state()
    assert len(state) == 625
    first = _core._rnorm(10)
    castorbind.set_rng_state(state)
    assert _core._rnorm(10) == first


@pytest.mark.parametrize("seed", [-(2**31), 2**31])
def test_set_seed_out_of_range(seed):
    with pytest.raises(ValueError):
        castorbind.set_seed(seed)


def test_set_rng_state_invalid():
    castorbind.set_seed(3)
    state = castorbind.get_rng_state()
    with pytest.raises(ValueError):
        castorbind.set_rng_state([0] * 625)
    with pytest.raises(ValueError):
        castorbind.set_rng_state(state[:-1])
    assert castorbind.get_rng_state() == state


@pytest.mark.parametrize("seed", [0, 1, 42, -12345, 2**31 - 1])
def test_seeding_matches_r(r, seed):
    castorbind.set_seed(seed)
    expected = r.eval(f"set.seed({seed}); .Random.seed[-1]")
    assert castorbind.get_rng_state() == expected


@pytest.mark.parametrize(
    "draw, r_code",
    [
        (lambda: _core._runif(1000), "runif(1000)"),
        (lambda: _core._rnorm(1000), "rnorm(1000)"),
        (lambda: _core._rexp(1000), "rexp(1000)"),
        (lambda: _core._rbinom(100, 10, 0.5), "rbinom(100, 10, 0.5)"),
        (lambda: _core._rbinom(100, 1000, 0.3), "rbinom(100, 1000, 0.3)"),
    ],
)
def test_draws_match_r(r, draw, r_code):
    castorbind.set_seed(1)
    assert_matches_r(draw(), r.eval(f"set.seed(1); {r_code}"))


def test_state_after_draws_matches_r(r):
    castorbind.set_seed(5)
    _core._runif(700)  # past one Mersenne-Twister block
    expected = r.eval("set.seed(5); runif(700); .Random.seed[-1]")
    assert castorbind.get_rng_state() == expected


def test_set_state_from_r(r):
    state = r.eval("set.seed(9); runif(3); .Random.seed[-1]")
    castorbind.set_rng_state(state)
    assert_matches_r(_core._runif(5), r.eval("set.seed(9); runif(8)[4:8]"))
