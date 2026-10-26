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
    description = CASTOR / "DESCRIPTION"
    if not description.exists():
        pytest.skip("vendor/castor not checked out")
    (version,) = re.findall(
        r"^Version:\s*(\S+)$", description.read_text(), re.M
    )
    assert castorbind.castor_version == version


def test_every_export_is_bound():
    exports = castor_exports()
    assert exports
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


def test_castor_output_goes_to_sys_stdout(tmp_path, capsys):
    path = tmp_path / "distances.tsv"
    path.write_text("a\tb\t1.5\na\tc\t2.5\n")
    result = castorbind.read_distances_list(
        file_path=str(path),
        delimiter="\t",
        comment_prefix="#",
        header=False,
        distances_column=2,
        min_distance=-1.0,
        max_distance=10.0,
        max_Nedges=-1,
        verbose_interval=1,
        verbose_prefix="castor: ",
    )
    assert result["success"], result
    assert "castor: Read 2 lines so far" in capsys.readouterr().out


class _Interrupt(Exception):
    pass


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "setitimer"), reason="needs setitimer"
)
def test_signal_interrupts_castor():
    import signal
    import time

    def handler(signum, frame):
        raise _Interrupt

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    start = time.monotonic()
    try:
        with pytest.raises(_Interrupt):
            # takes several seconds if not interrupted
            castorbind.generate_random_tree(
                max_tips=50_000_000,
                max_extant_tips=-1,
                max_time=-1.0,
                max_time_since_equilibrium=-1.0,
                birth_rate_intercept=1.0,
                birth_rate_factor=0.0,
                birth_rate_exponent=1.0,
                death_rate_intercept=0.0,
                death_rate_factor=0.0,
                death_rate_exponent=1.0,
                additional_rates_times=[],
                additional_birth_rates_pc=[],
                additional_death_rates_pc=[],
                additional_periodic=False,
                coalescent=False,
                Nsplits=2,
                as_generations=False,
                no_full_extinction=True,
                include_birth_times=False,
                include_death_times=False,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.monotonic() - start < 5
