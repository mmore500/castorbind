"""Run genuine R and castor on the fly, for comparison with castorbind.

R is found from $CASTORBIND_RSCRIPT, else Rscript on the PATH (see
tests/install_r_env.py). Tests that need R are skipped when it, or the
matching castor version, is unavailable, unless $CASTORBIND_REQUIRE_R is
set, in which case they fail.
"""

import json
import math
import os
import shutil
import subprocess
import typing

import numpy as np
import pytest

import castorbind

# serialize R values as JSON; doubles round-trip exactly via %.17g
R_PRELUDE = r"""
suppressPackageStartupMessages(library(castor))
.tojson <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    items <- vapply(x, .tojson, "")
    if (is.null(names(x))) return(paste0("[", paste(items, collapse=","), "]"))
    keys <- paste0("\"", names(x), "\":")
    return(paste0("{", paste0(keys, items, collapse=","), "}"))
  }
  if (is.matrix(x)) {
    rows <- vapply(seq_len(nrow(x)), function(i) .tojson(x[i, ]), "")
    return(paste0("[", paste(rows, collapse=","), "]"))
  }
  if (is.character(x)) {
    vals <- paste0("\"", gsub("([\"\\\\])", "\\\\\\1", x), "\"")
  } else if (is.logical(x)) {
    vals <- ifelse(is.na(x), "NaN", ifelse(x, "true", "false"))
  } else if (is.complex(x)) {
    stop("complex results are not supported")
  } else {
    x <- as.numeric(x)
    vals <- ifelse(is.nan(x) | is.na(x), "NaN",
      ifelse(is.infinite(x), ifelse(x > 0, "Infinity", "-Infinity"),
        sprintf("%.17g", x)))
  }
  paste0("[", paste(vals, collapse=","), "]")
}
"""


def to_r_literal(value: typing.Any) -> str:
    """Python argument -> equivalent R expression."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{float(value):.17g}"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        items = (f"{k}={to_r_literal(v)}" for k, v in value.items())
        return f"list({', '.join(items)})"
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (list, tuple, np.ndarray)) for v in value
    ):
        return f"list({', '.join(to_r_literal(v) for v in value)})"
    values = list(np.asarray(value).ravel())
    if not values:
        return "numeric()"
    if isinstance(values[0], str):
        return f"c({', '.join(json.dumps(v) for v in values)})"
    return f"c({', '.join(to_r_literal(v) for v in values)})"


class RSession:
    def __init__(self, rscript: str) -> None:
        self.rscript = rscript

    def eval(self, code: str) -> typing.Any:
        """Evaluate R code; its last value is returned via JSON."""
        script = f"{R_PRELUDE}\n.result <- local({{\n{code}\n}})\n"
        script += "cat(.tojson(.result))\n"
        completed = subprocess.run(
            [self.rscript, "--vanilla", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"R failed:\n{completed.stderr}")
        return json.loads(completed.stdout)

    def call_castor(
        self, name: str, seed: typing.Optional[int] = None, **kwargs
    ) -> typing.Any:
        """Call castor:::<name>_CPP in R with the same arguments."""
        args = ", ".join(f"{k}={to_r_literal(v)}" for k, v in kwargs.items())
        setup = "" if seed is None else f"set.seed({seed})\n"
        return self.eval(f"{setup}castor:::{name}_CPP({args})")


def _find_rscript() -> typing.Optional[str]:
    return os.environ.get("CASTORBIND_RSCRIPT") or shutil.which("Rscript")


@pytest.fixture(scope="session")
def r() -> RSession:
    def unavailable(reason: str) -> None:
        if os.environ.get("CASTORBIND_REQUIRE_R"):
            pytest.fail(reason)
        pytest.skip(reason)

    rscript = _find_rscript()
    if rscript is None:
        unavailable("Rscript not found; see tests/install_r_env.py")
    session = RSession(rscript)
    try:
        (version,) = session.eval('as.character(packageVersion("castor"))')
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        unavailable(f"R with castor unavailable: {e}")
    if version != castorbind.castor_version:
        unavailable(
            f"R has castor {version}, castorbind binds "
            f"{castorbind.castor_version}"
        )
    return session


def assert_matches_r(actual: typing.Any, expected: typing.Any) -> None:
    """Compare a castorbind result with its JSON-ified R counterpart.

    R has no scalars, so a length-1 R vector matches a python scalar.
    Numbers must match exactly (NaN matches NaN).
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), actual
        assert actual.keys() == expected.keys()
        for key in expected:
            try:
                assert_matches_r(actual[key], expected[key])
            except AssertionError as e:
                raise AssertionError(f"in [{key!r}]: {e}") from e
        return
    if isinstance(actual, dict):
        raise AssertionError(f"expected {expected!r}, got dict")
    if not isinstance(actual, (list, np.ndarray)):
        actual = [actual]
    if expected and isinstance(expected[0], str):
        assert list(actual) == expected
    elif (
        expected
        and isinstance(expected[0], list)
        and not isinstance(actual, np.ndarray)
    ):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_matches_r(a, e)
    else:
        np.testing.assert_array_equal(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float).reshape(np.shape(actual)),
        )
