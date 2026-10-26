"""Argument validation at the Python boundary."""

from decimal import Decimal
import math

import numpy as np
import pytest

from castorbind import legacy as cb

# pybind11 converts scalar and string arguments, and its errors list the
# whole signature rather than naming the bad argument; each rejection test
# below has an accepted counterpart differing only in that argument
INCOMPATIBLE = "incompatible function arguments"

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
    with pytest.raises(TypeError, match=INCOMPATIBLE):
        distance_matrix(verbose=value)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_bool_accepted(value):
    distance_matrix(verbose=value)


@pytest.mark.parametrize(
    "value",
    [
        math.inf,
        -math.inf,
        math.nan,
        1e300,
        2.5,
        3.0,
        2**70,
        np.float32(3.7),
        np.float64(3.0),
        Decimal("3.9"),
        None,
    ],
)
def test_integer_rejects_non_ints(value):
    with pytest.raises(TypeError, match=INCOMPATIBLE):
        cb.get_distances_from_root(**dict(TREE, Ntips=value))


def test_integer_accepts_numpy_int():
    assert cb.get_distances_from_root(**dict(TREE, Ntips=np.int64(3)))


def test_narrow_integer_out_of_range():
    # quoting is an `int` in castor
    with pytest.raises(TypeError, match=INCOMPATIBLE):
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


@pytest.mark.parametrize("labels", [None, "a", [b"a", 1, "c"]])
def test_string_list_rejects_non_lists(labels):
    kwargs = dict(
        TREE,
        tip_labels=["a", "b", "c"],
        node_labels=[],
        edge_labels=[],
        edge_numbers=[],
        digits=10,
        root_edge_length=-1.0,
        quoting=0,
    )
    assert cb.tree_to_Newick_string(**kwargs)
    with pytest.raises(TypeError, match=INCOMPATIBLE):
        cb.tree_to_Newick_string(**dict(kwargs, node_labels=labels))


def test_list_of_numpy_arrays():
    # each array is one row, as with nested python lists
    expected = cb.dense_binary_matrix_row2column_major(
        NR=2, NC=4, dense_rows=[[0, 2], [1, 3]], Rindexing=False
    )
    actual = cb.dense_binary_matrix_row2column_major(
        NR=2,
        NC=4,
        dense_rows=[np.array([0, 2]), np.array([1, 3])],
        Rindexing=False,
    )
    assert actual == expected
    assert cb.dense_binary_matrix_row2column_major(
        NR=2,
        NC=4,
        dense_rows=[np.array([0, 2]), np.array([1])],
        Rindexing=False,
    )


@pytest.mark.parametrize("row", [[1, None], [1, "a"], {1, 2}])
def test_list_rejects_non_numbers(row):
    with pytest.raises(TypeError, match="list elements must be"):
        cb.dense_binary_matrix_row2column_major(
            NR=1, NC=2, dense_rows=[row], Rindexing=False
        )


def test_list_rejects_complex():
    with pytest.raises(NotImplementedError, match="complex"):
        cb.dense_binary_matrix_row2column_major(
            NR=1, NC=2, dense_rows=[[1j]], Rindexing=False
        )


def test_missing_list_name_is_key_error():
    functor = cb.SBM_get_SBM_PD_functor(max_error=1e-3, max_Legendre_terms=100)
    del functor["approx_max_tD"]
    with pytest.raises(KeyError, match="approx_max_tD"):
        cb.TSBM_LL_of_transitions(
            radius=1.0,
            MRCA_times=[0.0],
            child_times1=[1.0],
            child_times2=[1.0],
            distances=[0.1],
            time_grid=[0.0, 1.0],
            diffusivities=[1.0, 1.0],
            splines_degree=1,
            SBM_PD_functor=functor,
        )


def test_nested_numpy_arrays_not_flattened():
    # a row that is itself a list of arrays is a list, as with python lists,
    # so castor rejects it instead of seeing one flattened row
    for row in [[[0, 2], [1, 3]], [np.array([0, 2]), np.array([1, 3])]]:
        with pytest.raises(ValueError, match="numeric"):
            cb.dense_binary_matrix_row2column_major(
                NR=1, NC=4, dense_rows=[row], Rindexing=False
            )
