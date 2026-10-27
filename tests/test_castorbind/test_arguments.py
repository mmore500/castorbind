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
    np.testing.assert_equal(
        cb.get_distances_from_root(**dict(TREE, Ntips=np.int64(3))),
        cb.get_distances_from_root(**TREE),
    )


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


def test_list_argument_rejects_str():
    with pytest.raises(TypeError, match="dense_rows"):
        cb.dense_binary_matrix_row2column_major(
            NR=1, NC=2, dense_rows="ab", Rindexing=False
        )


@pytest.mark.parametrize("mti", [-5, 626, 2**31])
def test_set_rng_state_rejects_bad_position(mti):
    state = cb.get_rng_state()
    with pytest.raises(ValueError):
        cb.set_rng_state([mti] + state[1:])
    assert cb.get_rng_state() == state


def test_matrix_result_layout():
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


def test_matrix_arguments_not_supported():
    # only find_non_zeros_float/int take matrices; use np.nonzero instead
    with pytest.raises(NotImplementedError, match="matrix"):
        cb.find_non_zeros_float(2, 2, np.eye(2))


@pytest.mark.parametrize("rows", [[[0, 2], [1]], ([0, 1],)])
def test_list_arguments_not_supported(rows):
    with pytest.raises(NotImplementedError, match="list arguments"):
        cb.dense_binary_matrix_row2column_major(
            NR=len(rows), NC=4, dense_rows=rows, Rindexing=False
        )


def sbm_functor():
    return cb.SBM_get_SBM_PD_functor(max_error=1e-3, max_Legendre_terms=100)


def tsbm_ll(functor):
    return cb.TSBM_LL_of_transitions(
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


def test_dict_argument_round_trip():
    # castor's own dict results, nested dicts included, go back in
    assert tsbm_ll(sbm_functor())["success"]


@pytest.mark.parametrize("value", ["a", [1, None], {1, 2}])
def test_dict_values_must_be_numeric(value):
    functor = sbm_functor()
    functor["approx_max_tD"] = value
    with pytest.raises(TypeError, match="dict values must be"):
        tsbm_ll(functor)


@pytest.mark.parametrize("value", [[[1, 2], [3]], [np.array([1.0]), [2.0]]])
def test_dict_values_nested_lists_not_supported(value):
    functor = sbm_functor()
    functor["approx_max_tD"] = value
    with pytest.raises(NotImplementedError, match="lists of lists"):
        tsbm_ll(functor)


def test_complex_arguments_not_supported():
    kwargs = dict(
        TREE,
        Nstates=2,
        transition_matrix=[-0.3, 0.3, 0.1, -0.1],
        eigenvalues=[],
        EVmatrix=[],
        inverse_EVmatrix=[],
        prior_probabilities_per_tip=[1, 0, 0, 1, 0.5, 0.5],
        root_prior_type="likelihoods",
        root_prior=[],
        include_ancestral_likelihoods=False,
        reroot=True,
        runtime_out_seconds=0.0,
        exponentiation_accuracy=1e-4,
        max_polynomials=1000,
        store_exponentials=True,
    )
    assert cb.ASR_with_fixed_rates_Markov_model(**kwargs)["success"]
    with pytest.raises(NotImplementedError, match="eigenvalues"):
        cb.ASR_with_fixed_rates_Markov_model(
            **dict(kwargs, eigenvalues=[0.0, -0.4])
        )
