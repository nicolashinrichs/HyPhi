"""
Regression tests for the FRC-weighted Graph Diffusion Distance (GDD) chain.

This pins the end-to-end pipeline that the retired ``GDD_FRc_*`` notebooks were
the only executable demonstration of: compute Forman-Ricci curvature bundles,
fit ONE global positive linear transform of the (signed) curvatures, attach the
resulting strictly-positive weights to each graph, and compute graph diffusion
distance via the closed-form heat kernel.

The pipeline lives in :mod:`hyphi.modeling.transform_curvature` and
:mod:`hyphi.modeling.GDD_FRc_helpers` (the heat-kernel maths in
:mod:`hyphi.modeling.curvatures`). These tests fix the load-bearing invariants
of the curvature transform (sign flip, ordering, strict positivity) and the GDD
matrix (symmetry, zero self-distance) on small deterministic graphs.

Note: this chain is the closed-form GDD used by the notebooks; it is independent
of the separate, time-swept :func:`hyphi.spectral.diffusion_distance.diffusion_distance`
(the crash tracked in issue #106), which has no callers and is not exercised here.
"""

import numpy as np
import pytest
from hyphi.modeling.GDD_FRc_helpers import (
    compute_frc_bundles_from_adjacencies,
    compute_pairwise_gdd_matrix,
    compute_successive_gdd,
)
from hyphi.modeling.transform_curvature import (
    apply_linear_transform,
    attach_edge_weights_to_graph,
    check_all_positive,
    fit_global_positive_linear_transform,
    transform_curvature_collection,
)

# Three small deterministic 4-node graphs with distinct topologies: a path-ish
# graph with a pendant, a 4-cycle, and the complete graph K4.
_A_PENDANT = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
_A_CYCLE = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)
_A_K4 = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)


def _build_weighted_graphs(adjacencies):
    """Run the full transform chain and return the positively-weighted graphs and the fit params."""
    bundles = compute_frc_bundles_from_adjacencies(adjacencies, method="1d")
    edge_curvatures = {idx: bundle["edge_curvatures"] for idx, bundle in bundles.items()}
    params = fit_global_positive_linear_transform(edge_curvatures)
    weights = transform_curvature_collection(edge_curvatures, params)
    weighted = {idx: attach_edge_weights_to_graph(bundles[idx]["graph"], weights[idx]) for idx in adjacencies}
    return weighted, params, weights


def test_global_transform_is_sign_flip_with_unit_scale():
    """The default fit flips the sign (sign=-1) and does not rescale (scale=1)."""
    adjacencies = {1: _A_PENDANT, 2: _A_CYCLE, 3: _A_K4}
    bundles = compute_frc_bundles_from_adjacencies(adjacencies, method="1d")
    edge_curvatures = {idx: bundle["edge_curvatures"] for idx, bundle in bundles.items()}

    params = fit_global_positive_linear_transform(edge_curvatures)

    assert params["sign"] == -1.0
    assert params["scale"] == pytest.approx(1.0)


def test_transformed_weights_are_strictly_positive():
    """Strict-positivity invariant: every transformed edge weight is > 0 across all graphs."""
    adjacencies = {1: _A_PENDANT, 2: _A_CYCLE, 3: _A_K4}
    _, _, weights = _build_weighted_graphs(adjacencies)

    positivity = check_all_positive(weights)
    assert all(entry["all_positive"] for entry in positivity.values())
    for per_graph in weights.values():
        assert all(value > 0.0 for value in per_graph.values())


def test_transform_flips_ordering_so_more_negative_curvature_gets_larger_weight():
    """The sign flip means a more negative curvature maps to a larger (more conductive) weight."""
    params = fit_global_positive_linear_transform({1: [-3.0, 0.0, 2.0]})
    low_curv = apply_linear_transform([-3.0], sign=params["sign"], scale=params["scale"], shift=params["shift"])[0]
    high_curv = apply_linear_transform([2.0], sign=params["sign"], scale=params["scale"], shift=params["shift"])[0]

    assert low_curv > high_curv
    assert high_curv > 0.0


def test_pairwise_gdd_matrix_is_symmetric_with_zero_self_distance():
    """The pairwise GDD matrix is symmetric, non-negative, finite, and zero on the diagonal."""
    adjacencies = {1: _A_PENDANT, 2: _A_CYCLE, 3: _A_K4}
    weighted, _, _ = _build_weighted_graphs(adjacencies)

    indices, distance_matrix, laplacians = compute_pairwise_gdd_matrix(weighted)

    assert indices == [1, 2, 3]
    assert set(laplacians) == {1, 2, 3}
    assert np.allclose(distance_matrix, distance_matrix.T)
    assert np.allclose(np.diag(distance_matrix), 0.0)
    assert np.all(np.isfinite(distance_matrix))
    assert np.all(distance_matrix >= 0.0)
    # Distinct topologies are distinguishable: every off-diagonal distance is positive.
    off_diagonal = distance_matrix[~np.eye(3, dtype=bool)]
    assert np.all(off_diagonal > 0.0)


def test_pairwise_gdd_matches_known_heat_kernel_values():
    """Pin the actual heat-kernel distances (not just the matrix bookkeeping) to known values.

    These values are computed by :func:`hyphi.modeling.curvatures.heat_kernel_distance` on the
    transformed-weight Laplacians of the three fixed graphs; they guard against a silent change
    in the heat-kernel maths. The tolerance is loose enough to be stable across the eigendecomposition
    backends on the CI Python matrix.
    """
    adjacencies = {1: _A_PENDANT, 2: _A_CYCLE, 3: _A_K4}
    weighted, _, _ = _build_weighted_graphs(adjacencies)

    _, distance_matrix, _ = compute_pairwise_gdd_matrix(weighted)

    assert distance_matrix[0, 1] == pytest.approx(1.41421349, abs=1e-3)  # pendant vs 4-cycle
    assert distance_matrix[0, 2] == pytest.approx(1.17684454, abs=1e-3)  # pendant vs K4
    assert distance_matrix[1, 2] == pytest.approx(1.73205079, abs=1e-3)  # 4-cycle vs K4


def test_successive_gdd_keys_and_zero_distance_for_identical_graphs():
    """Successive GDD is keyed by the earlier index and is exactly zero for identical graphs."""
    adjacencies = {1: _A_PENDANT, 2: _A_CYCLE, 3: _A_K4}
    weighted, _, _ = _build_weighted_graphs(adjacencies)

    successive = compute_successive_gdd(weighted)
    assert set(successive) == {1, 2}
    assert all(np.isfinite(value) and value >= 0.0 for value in successive.values())

    # Known answer: the diffusion distance of a graph from itself is zero.
    duplicated = {1: weighted[1], 2: weighted[1]}
    assert compute_successive_gdd(duplicated)[1] == pytest.approx(0.0, abs=1e-9)


def test_successive_gdd_requires_at_least_two_graphs():
    """A single graph cannot form a successive distance and raises ValueError."""
    # The pendant graph (unlike vertex-transitive K4) has curvature spread, so the
    # global fit succeeds and we reach the single-graph guard in compute_successive_gdd.
    adjacencies = {1: _A_PENDANT}
    weighted, _, _ = _build_weighted_graphs(adjacencies)

    with pytest.raises(ValueError, match="at least two graphs"):
        compute_successive_gdd(weighted)


def test_fit_rejects_degenerate_and_empty_inputs():
    """Fitting rejects an empty collection and a collection with no curvature spread."""
    with pytest.raises(ValueError, match="non-empty dictionary"):
        fit_global_positive_linear_transform({})

    with pytest.raises(ValueError, match="identical"):
        fit_global_positive_linear_transform({1: [2.0, 2.0, 2.0]})
