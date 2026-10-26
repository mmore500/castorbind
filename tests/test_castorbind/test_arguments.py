"""Argument validation at the Python boundary."""

import math

import numpy as np
import pytest

from castorbind import legacy as cb

TREE = {
    "Ntips": 3,
    "Nnodes": 2,
    "Nedges": 4,
    "tree_edge": [3, 0, 3, 4, 4, 1, 4, 2],
    "edge_length": [2.0, 1.0, 1.0, 1.0],
}


def distance_matrix(**overrides):
    kwargs = dict(TREE, focal_clades=[0, 1], verbose=False, verbose_prefix="")
    kwargs.update(overrides)
    return cb.get_distance_matrix_between_clades(**kwargs)


@pytest.mark.parametrize("value", [None, 0, 1, 2, 0.3, "yes", np.array([1])])
def test_bool_is_strict(value):
    with pytest.raises(TypeError, match="verbose"):
        distance_matrix(verbose=value)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_bool_accepted(value):
    distance_matrix(verbose=value)


@pytest.mark.parametrize(
    "value", [math.inf, -math.inf, math.nan, 1e300, 2.5, 3.0, 2**70]
)
def test_integer_rejects_non_ints(value):
    with pytest.raises(TypeError, match="Ntips"):
        cb.get_distances_from_root(**dict(TREE, Ntips=value))


def test_integer_accepts_numpy_int():
    assert cb.get_distances_from_root(**dict(TREE, Ntips=np.int64(3)))


def test_narrow_integer_out_of_range():
    # quoting is an `int` in castor
    with pytest.raises(TypeError, match="quoting"):
        cb.tree_to_Newick_string(
            **TREE,
            tip_labels=["a", "b", "c"],
            node_labels=[],
            edge_labels=[],
            edge_numbers=[],
            digits=10,
            root_edge_length=-1.0,
            quoting=2**40,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.5])
def test_integer_vector_rejects_bad_floats(bad):
    tree_edge = np.array(TREE["tree_edge"], dtype=float)
    tree_edge[1] = bad
    with pytest.raises(TypeError, match="tree_edge"):
        cb.get_distances_from_root(**dict(TREE, tree_edge=tree_edge))


def test_integer_matrix_rejects_non_integral():
    with pytest.raises(TypeError, match="A"):
        cb.find_non_zeros_int(1, 2, np.array([[0.5, 1.9]]))
    result = cb.find_non_zeros_int(1, 2, np.array([[0.0, 2.0]]))
    assert result == {"rows": [0], "columns": [1], "values": [2]}


def test_list_argument_rejects_str():
    with pytest.raises(TypeError, match="dense_rows"):
        cb.dense_binary_matrix_row2column_major(
            NR=1, NC=2, dense_rows="ab", Rindexing=False
        )


def test_list_of_dicts():
    # list elements that are dicts become R lists; castor's helper doesn't
    # take such input, so just check it reaches castor's own error path
    with pytest.raises(Exception) as e:
        cb.dense_binary_matrix_row2column_major(
            NR=2, NC=2, dense_rows=[{"a": 1}, {"b": 2}], Rindexing=False
        )
    assert "cannot convert" not in str(e.value)


@pytest.mark.parametrize("mti", [-5, 626, 2**31])
def test_set_rng_state_rejects_bad_position(mti):
    state = cb.get_rng_state()
    with pytest.raises(ValueError):
        cb.set_rng_state([mti] + state[1:])
    assert cb.get_rng_state() == state


def test_matrix_result_layout():
    result = cb.find_non_zeros_float(2, 3, np.arange(6.0).reshape(2, 3))
    assert result["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    distances = distance_matrix(focal_clades=[0, 1, 2, 3])
    assert distances.flags.c_contiguous
    assert distances[0, 3] == 2.0 and distances[3, 0] == 2.0
