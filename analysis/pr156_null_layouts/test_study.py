"""Independent finite-support and metadata checks for the research comparators."""

import itertools

import numpy as np
import pytest
from study import (
    MODES,
    SCENARIOS,
    Scenario,
    compare,
    summary,
    evaluate,
    naive_map,
    production,
    references,
    assignments,
    make_layout,
)


@pytest.mark.parametrize("scenario", SCENARIOS[:-1], ids=lambda s: s.name)
def test_enumerated_production_matches_naive_and_identity(scenario):
    """Every small-support A value agrees with direct list sorting; identities agree."""
    data = make_layout(scenario, 7)
    pm = assignments(data)
    actual = production._group_maps_np(data["layout"], pm)
    expected, _ = evaluate(data, pm, "A")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    identity = np.arange(data["layout"].n)
    observed = production._group_maps_np(data["layout"], identity[None])[0]
    for mode in MODES:
        assert naive_map(data, identity, mode)[0] == pytest.approx(observed, abs=1e-14)


@pytest.mark.parametrize(
    "name", ["shared_exchangeable", "same_plate_zero", "same_plate_strong"]
)
def test_invariant_reference_sets(name):
    """No restrictions or within-plate pools leave A's reference sets unchanged."""
    data = make_layout(next(s for s in SCENARIOS if s.name == name), 19)
    for pm in assignments(data):
        for slot in range(len(pm)):
            assert set(references(data, pm, slot, "A")) == set(
                references(data, pm, slot, "B_local")
            )
            assert set(references(data, pm, slot, "A")) == set(
                references(data, pm, slot, "B_global")
            )


def test_control_roles_move_and_can_swap_completely():
    """Old controls become members; old members become controls, not frozen flags."""
    data = make_layout(Scenario("swap", (2, 0), (0, 2), "different"), 7)
    pm = np.array([2, 3])
    for mode in MODES:
        assert references(data, pm, 0, mode) == [0, 1]
        assert references(data, pm, 1, mode) == [0, 1]
    # Mixed labels have no original negative edge that satisfies the fixed rule.
    mixed = np.array([0, 2])
    assert references(data, mixed, 0, "B_local") == [3]
    assert references(data, mixed, 1, "B_local") == [1]
    assert references(data, mixed, 0, "A") == [1, 3]


def test_global_reapplication_adds_other_block_references():
    """A hand-specified two-block draw separates local filtering from global rebuilding."""
    data = make_layout(Scenario("two_blocks", (1, 1), (1, 1), "different"), 7)
    original_to_code = {int(p): i for i, p in enumerate(data["profiles"])}
    # Original rows: members plate 0,1; controls plate 0,1.
    # Keep member 0 but replace member 1 by control 2 (both now on plate 0).
    pm = np.array([original_to_code[0], original_to_code[2]])
    for slot in range(2):
        local_original = {
            int(data["profiles"][r]) for r in references(data, pm, slot, "B_local")
        }
        global_original = {
            int(data["profiles"][r]) for r in references(data, pm, slot, "B_global")
        }
        assert local_original == ({3} if slot == 0 else {1})
        assert global_original == {1, 3}
        assert global_original > local_original


def test_block_counts_and_complete_positive_slot_invariance():
    """Exact and production assignments preserve original block counts, not plate counts."""
    data = make_layout(SCENARIOS[6], 41)
    layout = data["layout"]
    drawn = production._draws_np(
        layout, 0, 512, *production.group_key(156, data["group"][0])
    )
    for pm in itertools.chain(assignments(data), drawn):
        assert len(set(pm)) == layout.n
        for b, pool in enumerate(data["pools"]):
            slots = np.flatnonzero(layout.slot_block == b)
            assert set(pm[slots]) <= set(pool)
            assert sum(x in pool for x in pm) == len(slots)
        shuffled = pm.copy()
        for b in range(len(data["pools"])):
            slots = np.flatnonzero(layout.slot_block == b)
            shuffled[slots] = pm[slots[::-1]]
        for mode in MODES:
            assert naive_map(data, pm, mode)[0] == pytest.approx(
                naive_map(data, shuffled, mode)[0]
            )


def test_hand_calculated_ap_and_positive_first_ties():
    """A rank list with one positive has AP 1/rank, including the declared tie policy."""
    data = make_layout(Scenario("hand", (2,), (2,), "none"), 7)
    # Both members rank negative 2, positive, negative 3: AP=1/2 each.
    data["keys"] = np.array(
        [[0, 2, 1, 3], [2, 0, 1, 3], [1, 1, 0, 2], [3, 3, 2, 0]], dtype=np.float32
    )
    for mode in MODES:
        assert naive_map(data, np.array([0, 1]), mode) == (0.5, [2, 2])
    data["keys"][:] = 1
    for mode in MODES:
        assert naive_map(data, np.array([0, 1]), mode)[0] == 1.0


def test_filter_monotonicity_and_zero_negative_slots():
    """Local filtering cannot reduce AP; zero-negative slots are kept with AP=1."""
    data = make_layout(Scenario("zeros", (3, 0), (0, 2), "different"), 7)
    pm = assignments(data)
    a, ac = evaluate(data, pm, "A")
    b, bc = evaluate(data, pm, "B_local")
    assert np.all(b >= a)
    assert np.all(bc <= ac)
    assert np.any(bc == 0)
    # Select the two controls and one original member: selected old member has
    # no unselected opposite-plate controls, but still two positives.
    chosen = np.array([0, 3, 4])
    assert references(data, chosen, 0, "B_local") == []
    value, counts = naive_map(data, chosen, "B_local")
    assert counts == [0, 2, 2]
    assert np.isfinite(value)


def test_degenerate_identity_and_empty_support():
    """No-reference pools give a single labelling; absent supports have no invented p."""
    data = make_layout(Scenario("identity", (2,), (0,), "none"), 7)
    pm = assignments(data)
    assert pm.shape == (1, 2)
    for mode in MODES:
        values, counts = evaluate(data, pm, mode)
        assert values.tolist() == [1.0]
        assert counts.tolist() == [[0, 0]]
    assert summary(np.array([]), 1.0) == {"status": "empty_support", "n": 0}
    with pytest.raises(ValueError, match="positive graph"):
        make_layout(Scenario("undefined", (1,), (2,), "none"), 7)


def test_overlapping_references_are_explicitly_unsupported():
    """Three-plate cross-reference layouts fail rather than silently switching nulls."""
    with pytest.raises(ValueError, match="overlapping but different references"):
        make_layout(SCENARIOS[-1], 7)


def test_exact_and_paired_mc_tail_conventions():
    """Finite-support tails and plus-one sampled tails use different denominators."""
    a = np.array([0.2, 0.4, 0.8, 0.8])
    b = np.array([0.4, 0.8, 0.8, 1.0])
    assert summary(a, 0.8)["p"] == 0.5
    assert summary(a, 0.8, sampled=True)["p"] == 0.6
    assert compare(a, b, 0.8)["p_difference"] == 0.25
    assert compare(a, b, 0.8, sampled=True)["p_difference"] == 0.2
    assert compare(a, a, 0.8)["wasserstein"] == 0
