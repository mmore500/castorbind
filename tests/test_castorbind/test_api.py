import inspect
import pathlib
import re
import subprocess
import sys

import pytest

import castorbind

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASTOR = ROOT / "vendor" / "castor"


def castor_exports():
    exports = CASTOR / "src" / "RcppExports.cpp"
    if not exports.exists():
        pytest.skip("vendor/castor not checked out")
    return re.findall(r"^// (\w+)_CPP$", exports.read_text(), re.M)


def test_castor_version():
    assert castorbind.castor_version == "1.8.7"
    assert castorbind.__version__.startswith(castorbind.castor_version)


def test_every_export_is_bound():
    exports = castor_exports()
    assert len(exports) == 174
    for name in exports:
        assert callable(getattr(castorbind, name)), name
        assert name in castorbind.__all__


def test_rng_functions_exported():
    for name in "set_seed", "get_rng_state", "set_rng_state":
        assert name in castorbind.__all__


def test_default_arguments():
    result = castorbind.find_non_zeros_float(2, 2, [[0.0, 1.0], [0.0, 0.0]])
    assert result == {"rows": [0], "columns": [1], "values": [1.0]}
    result = castorbind.find_non_zeros_float(
        2, 2, [[0.0, 1.0], [0.0, 0.0]], transpose=True
    )
    assert result == {"rows": [1], "columns": [0], "values": [1.0]}


def test_docstrings():
    doc = inspect.getdoc(castorbind.get_distances_from_root)
    assert "get_distances_from_root_CPP" in doc


def test_bad_argument_type():
    with pytest.raises(TypeError, match="Ntips"):
        castorbind.get_distances_from_root(
            Ntips="three",
            Nnodes=2,
            Nedges=4,
            tree_edge=[3, 0, 3, 4, 4, 1, 4, 2],
            edge_length=[2.0, 1.0, 1.0, 1.0],
        )


def test_type_stubs_up_to_date():
    castor_exports()  # skip without vendor/castor
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "gen_bindings.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
