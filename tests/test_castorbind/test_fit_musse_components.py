"""Compare, against genuine R, the C++ functions castor's fit_musse uses.

fit_musse itself is R-level, so it isn't wrapped; these are the castor C++
routines it (and its start-value guesses and bootstraps) calls, with
arguments shaped the way castor's R code passes them.
"""

import pytest

from ..conftest import check

NEWICK = "((t0:1,t1:1):1,(t2:1.5,(t3:0.5,t4:0.5):1):0.5);"  # ultrametric
NSTATES = 2
# row-major transition matrix, as castor's R code flattens it
Q = [-0.3, 0.3, 0.1, -0.1]
TIP_STATES = [0, 1, 0, 1, 1]  # 0-based, per tip index


@pytest.fixture(scope="module")
def tree(r):
    tree = check(
        r,
        "read_Newick_string",
        input=NEWICK,
        underscores_as_blanks=False,
        interpret_quotes=True,
        look_for_edge_names=False,
        look_for_edge_numbers=False,
    )
    return {
        "Ntips": tree["Ntips"],
        "Nnodes": tree["Nnodes"],
        "Nedges": tree["Nedges"],
        "tree_edge": tree["tree_edge"],
        "edge_length": tree["edge_lengths"],
    }


@pytest.fixture(scope="module")
def root_age(r, tree):
    # fit_musse: root_age = get_tree_span(tree)$max_distance
    span = check(r, "get_min_max_tip_distance_from_root", **tree)
    return span["max_distance"]


@pytest.fixture(scope="module")
def clade_ages(r, tree, root_age):
    # fit_musse: clade_ages = root_age - get_all_distances_to_root(tree)
    distances = check(r, "get_distances_from_root", **tree)
    return [root_age - d for d in distances]


def tip_priors(Ntips, states, Nstates=NSTATES, unknown=()):
    """Row-major Ntips x Nstates likelihoods, flat for unknown tips."""
    priors = []
    for tip in range(Ntips):
        if tip in unknown:
            priors += [1.0 / Nstates] * Nstates
        else:
            priors += [float(s == states[tip]) for s in range(Nstates)]
    return priors


@pytest.mark.parametrize(
    "root_prior_type, root_prior",
    [
        ("custom", [0.3, 0.7]),
        ("likelihoods", []),
        ("max_likelihood", []),
    ],
)
@pytest.mark.parametrize(
    "root_conditioning", ["none", "madfitz", "herr_als", "crown", "stem"]
)
def test_musse_loglikelihood(
    r,
    tree,
    root_age,
    clade_ages,
    root_prior_type,
    root_prior,
    root_conditioning,
):
    result = check(
        r,
        "get_MuSSE_loglikelihood",
        ignore=("runtime",),  # wall-clock
        Ntips=tree["Ntips"],
        Nnodes=tree["Nnodes"],
        Nedges=tree["Nedges"],
        Nstates=NSTATES,
        oldest_age=root_age,
        tree_edge=tree["tree_edge"],
        clade_ages=clade_ages,
        transition_rates=Q,
        speciation_rates=[1.0, 1.5],
        extinction_rates=[0.2, 0.4],
        sampling_rates=[0.0, 0.0],
        initial_D_per_tip=tip_priors(tree["Ntips"], TIP_STATES, unknown=[3]),
        initial_E_per_state=[0.0, 0.0],
        root_prior_type=root_prior_type,
        root_prior=root_prior,
        root_conditioning=root_conditioning,
        include_ancestral_likelihoods=True,
        include_warnings=True,
        # fit_musse's defaults
        max_condition_number=1e4,
        relative_ODE_step=0.1,
        E_value_step=1e-4,
        D_temporal_resolution=100.0,
        runtime_out_seconds=0.0,
    )
    assert result["success"], result.get("error")


def test_musse_loglikelihood_three_states(r, tree, root_age, clade_ages):
    # incomplete sampling via initial E, and hidden-state-like tip priors
    result = check(
        r,
        "get_MuSSE_loglikelihood",
        ignore=("runtime",),
        Ntips=tree["Ntips"],
        Nnodes=tree["Nnodes"],
        Nedges=tree["Nedges"],
        Nstates=3,
        oldest_age=root_age,
        tree_edge=tree["tree_edge"],
        clade_ages=clade_ages,
        transition_rates=[-0.5, 0.2, 0.3, 0.1, -0.2, 0.1, 0.4, 0.4, -0.8],
        speciation_rates=[1.0, 2.0, 0.5],
        extinction_rates=[0.1, 0.5, 0.0],
        sampling_rates=[0.0, 0.0, 0.0],
        initial_D_per_tip=[
            0.5,
            0.5,
            0,
            0,
            0,
            0.5,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0.5,
        ],
        initial_E_per_state=[0.2, 0.2, 0.5],
        root_prior_type="likelihoods",
        root_prior=[],
        root_conditioning="madfitz",
        include_ancestral_likelihoods=True,
        include_warnings=True,
        max_condition_number=1e4,
        relative_ODE_step=0.1,
        E_value_step=1e-4,
        D_temporal_resolution=100.0,
        runtime_out_seconds=0.0,
    )
    assert result["success"], result.get("error")


def test_mk_loglikelihood(r, tree, root_age):
    # start-value guess via fit_mk
    check(
        r,
        "Mk_loglikelihood",
        ignore=("runtime",),
        **tree,
        Nstates=NSTATES,
        transition_matrix=Q,
        prior_probabilities_per_tip=tip_priors(tree["Ntips"], TIP_STATES),
        root_prior_type="likelihoods",
        root_prior=[],
        oldest_age=root_age,
        runtime_out_seconds=0.0,
        exponentiation_accuracy=1e-4,
        max_polynomials=1000,
    )


def test_max_parsimony_asr(r, tree):
    # start-value guess via asr_max_parsimony
    check(
        r,
        "WMPR_ASR",
        **tree,
        Nstates=NSTATES,
        tip_states=TIP_STATES,
        transition_costs=[0.0, 1.0, 1.0, 0.0],  # "all_equal"
        branch_length_exponent=0.0,
        weight_posteriors_by_scenario_counts=True,
        verbose=False,
        verbose_prefix="",
    )


def test_independent_contrasts(r, tree):
    # start-value guess via guesstimate_Mk_rate_via_independent_contrasts
    check(
        r,
        "get_phylogenetic_independent_contrasts",
        **tree,
        Ntraits=1,
        tip_states=[0.0, 1.0, 0.0, 1.0, 1.0],
        scaled=True,
        only_bifurcations=False,
        include_zero_phylodistances=False,
    )


def test_subtree_with_tips(r, tree):
    # get_subtree_with_tips, used by the start-value guesses
    check(
        r,
        "get_subtree_with_specific_clades",
        **tree,
        clades_to_keep=[0, 2, 4],
        collapse_monofurcations=True,
        force_keep_root=True,
        keep_all_children_of_explicit_clades_to_keep=False,
        keep_all_tips_of_explicit_clades_to_keep=False,
    )


def test_hbds_loglikelihood(r, root_age):
    # start-value guess via fit_hbds_model_on_grid
    result = check(
        r,
        "get_HBDS_model_loglikelihood",
        ignore=("runtime",),
        branching_ages=[0.5, 1.0, 1.5, 2.0],
        Ptip_ages=[],
        Pnode_ages=[],
        CSA_ages=[0.0],
        CSA_probs=[1.0],
        CSA_kappas=[0.0],
        concentrated_tip_counts=[5],
        concentrated_node_counts=[0],
        oldest_age=root_age,
        age_grid=[0.0, root_age],
        lambdas=[1.2, 1.0],
        mus=[0.2, 0.3],
        psis=[0.0, 0.0],
        kappas=[0.0, 0.0],
        splines_degree=1,
        condition="crown",
        relative_ODE_step=0.001,
        E_value_step=1e-4,
        runtime_out_seconds=0.0,
    )
    assert result["success"], result.get("error")


def test_deterministic_diversity_growth(r, root_age):
    # start-value guess via fit_tree_model
    result = check(
        r,
        "simulate_deterministic_diversity_growth",
        ignore=("runtime",),
        birth_rate_intercept=1.0,
        birth_rate_factor=0.0,
        birth_rate_exponent=1.0,
        death_rate_intercept=0.2,
        death_rate_factor=0.0,
        death_rate_exponent=1.0,
        resolution=0.0,
        rarefaction=1.0,
        Nsplits=2,
        additional_rates_times=[],
        additional_birth_rates_pc=[],
        additional_death_rates_pc=[],
        additional_periodic=False,
        times=[0.0, 0.5, 1.0, 1.5],
        start_time=0.0,
        final_time=root_age,
        start_diversity=1.0,
        final_diversity=5.0,
        reverse=True,
        include_coalescent=True,
        include_probabilities=True,
        include_birth_rates=True,
        include_death_rates=False,
        include_Nevents=False,
        runtime_out_seconds=0.0,
    )
    assert result["success"], result.get("error")


@pytest.mark.parametrize("seed", [1, 7])
def test_simulate_musse_tree(r, seed):
    # simulate_musse (used for fit_musse's bootstraps) -> simulate_dsse
    result = check(
        r,
        "generate_random_tree_Mk_rates",
        seed=seed,
        max_tips=30,
        max_extant_tips=-1,
        max_sampled_tips=-1,
        max_time=-1.0,
        max_time_since_equilibrium=-1.0,
        max_events=-1,
        Nstates=NSTATES,
        start_state=0,
        state_birth_rates=[1.0, 1.5],
        state_death_rates=[0.2, 0.4],
        state_sampling_rates=[0.0, 0.0],
        transition_matrix_A=Q,
        transition_matrix_C=[0.0] * 4,
        as_generations=False,
        no_full_extinction=True,
        include_extant=True,
        include_extinct=False,
        include_event_times=True,
        include_rates=True,
    )
    assert result["success"], result.get("error")
