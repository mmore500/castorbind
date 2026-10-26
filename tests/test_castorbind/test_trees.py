import numpy as np
import pytest

from castorbind import legacy as cb

NEWICK = "((a:1,b:1):1,c:2);"


def read(newick=NEWICK):
    return cb.read_Newick_string(
        input=newick,
        underscores_as_blanks=False,
        interpret_quotes=True,
        look_for_edge_names=False,
        look_for_edge_numbers=False,
    )


def test_read_newick():
    tree = read()
    assert tree["success"]
    assert tree["Ntips"] == 3
    assert tree["Nnodes"] == 2
    assert sorted(tree["tip_names"]) == ["a", "b", "c"]
    assert len(tree["tree_edge"]) == 2 * tree["Nedges"]


def test_read_newick_failure_reported():
    tree = read("((a:1,b:1")
    assert not tree["success"]
    assert tree["error"]


def test_distances_from_root():
    tree = read()
    depths = cb.get_distances_from_root(
        Ntips=tree["Ntips"],
        Nnodes=tree["Nnodes"],
        Nedges=len(tree["tree_edge"]) // 2,
        tree_edge=tree["tree_edge"],
        edge_length=tree["edge_lengths"],
    )
    assert depths[: tree["Ntips"]] == [2.0, 2.0, 2.0]
    assert depths[tree["root"]] == 0.0


@pytest.mark.parametrize("as_array", [False, True])
def test_newick_round_trip(as_array):
    tree = read()
    tree_edge = tree["tree_edge"]
    if as_array:  # (Nedges, 2) arrays flatten row-major, as castor expects
        tree_edge = np.array(tree_edge).reshape(-1, 2)
    newick = cb.tree_to_Newick_string(
        Ntips=tree["Ntips"],
        Nnodes=tree["Nnodes"],
        Nedges=tree["Nedges"],
        tree_edge=tree_edge,
        edge_length=tree["edge_lengths"],
        tip_labels=tree["tip_names"],
        node_labels=[],
        edge_labels=[],
        edge_numbers=[],
        digits=10,
        root_edge_length=-1,
        quoting=0,
    )
    again = read(newick)
    assert sorted(again["tip_names"]) == ["a", "b", "c"]
    assert sorted(again["edge_lengths"]) == sorted(tree["edge_lengths"])


def test_matrix_result():
    tree = read()
    distances = cb.get_distance_matrix_between_clades(
        Ntips=tree["Ntips"],
        Nnodes=tree["Nnodes"],
        Nedges=tree["Nedges"],
        tree_edge=tree["tree_edge"],
        edge_length=tree["edge_lengths"],
        focal_clades=[0, 1, 2],
        verbose=False,
        verbose_prefix="",
    )
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(distances), 0.0)
    np.testing.assert_array_equal(distances, distances.T)
    assert sorted(distances[0]) == [0.0, 4.0, 4.0]
