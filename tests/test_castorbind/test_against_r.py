"""Compare castorbind with castor's own C++, called from genuine R."""

import pytest

from ..conftest import check

TREE = dict(
    Ntips=5,
    Nnodes=4,
    Nedges=8,
    # ((t0:1,t1:2):0.5,(t2:1.5,(t3:0.25,t4:0.75):1):2);
    tree_edge=[5, 6, 6, 0, 6, 1, 5, 7, 7, 2, 7, 8, 8, 3, 8, 4],
    edge_length=[0.5, 1.0, 2.0, 2.0, 1.5, 1.0, 0.25, 0.75],
)


def test_read_newick(r):
    check(
        r,
        "read_Newick_string",
        input="((a:1,'b c':2.5)n1:0.5,d:3e-2)root;",
        underscores_as_blanks=False,
        interpret_quotes=True,
        look_for_edge_names=False,
        look_for_edge_numbers=False,
    )


def test_distances_from_root(r):
    check(r, "get_distances_from_root", **TREE)


def test_distance_matrix(r):
    check(
        r,
        "get_distance_matrix_between_clades",
        **TREE,
        focal_clades=[0, 2, 4, 6],
        verbose=False,
        verbose_prefix="",
    )


def test_tree_to_newick(r):
    check(
        r,
        "tree_to_Newick_string",
        **TREE,
        tip_labels=["t0", "t1", "t2", "t3", "t4"],
        node_labels=[],
        edge_labels=[],
        edge_numbers=[],
        digits=10,
        root_edge_length=-1.0,
        quoting=0,
    )


@pytest.mark.parametrize("seed", [1, 2024])
def test_generate_random_tree(r, seed):
    result = check(
        r,
        "generate_random_tree",
        seed=seed,
        max_tips=50,
        max_extant_tips=-1,
        max_time=-1.0,
        max_time_since_equilibrium=-1.0,
        birth_rate_intercept=1.0,
        birth_rate_factor=0.0,
        birth_rate_exponent=1.0,
        death_rate_intercept=0.2,
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
        include_birth_times=True,
        include_death_times=True,
    )
    assert result["success"]


def test_simulate_brownian_motion(r):
    check(
        r,
        "simulate_scalar_Brownian_motion_model",
        seed=3,
        **TREE,
        root_states=[0.0],
        diffusivity=1.5,
        include_tips=True,
        include_nodes=True,
        Nsimulations=4,
    )


def test_simulate_markov_model(r):
    check(
        r,
        "simulate_fixed_rates_Markov_model",
        seed=11,
        **TREE,
        Nstates=3,
        transition_matrix=[-1.0, 0.5, 0.5, 0.2, -0.4, 0.2, 1.0, 1.0, -2.0],
        root_probabilities=[0.2, 0.3, 0.5],
        include_tips=True,
        include_nodes=True,
        Nsimulations=5,
    )


def test_list_argument(r):
    # a ragged list of int vectors, passed as an R list
    check(
        r,
        "dense_binary_matrix_row2column_major",
        NR=3,
        NC=4,
        dense_rows=[[0, 2], [1], [0, 1, 3]],
        Rindexing=False,
    )
