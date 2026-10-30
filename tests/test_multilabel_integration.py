"""Essential integration checks for the fixed-draw multilabel null."""

import itertools

import numpy as np
import pandas as pd
import pytest

from copairs.map import (
    LabelPairs,
    get_map_pvalue,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from copairs.nulls import permutation
from copairs.map.multilabel import average_precision
from copairs.map.average_precision import attached_pairs


@pytest.fixture
def label_example():
    """Return overlapping labels whose A members have different reference sets."""
    meta = pd.DataFrame(
        {
            "target": [["a"], ["a", "b"], ["b"], ["c"]],
            "family": "f",
            "profile": range(4),
        }
    )
    feats = np.random.default_rng(17).normal(size=(4, 5))
    ap, pairs = average_precision(
        meta,
        feats,
        ["target"],
        [],
        [],
        ["target"],
        "target",
        progress_bar=False,
        return_pairs=True,
    )
    return meta, feats, ap, pairs


def test_expanded_row_mapping_and_joint(label_example):
    """Attach the expanded AP-row graph and use it in joint sampling."""
    _, _, ap, pairs = label_example
    assert isinstance(pairs, LabelPairs) and attached_pairs(ap) is pairs
    np.testing.assert_array_equal(pairs.profile[ap.index], ap.profile)
    np.testing.assert_array_equal(pairs.keys[pairs.row_key[ap.index]], ap.target)
    expected = np.array(
        [
            (a, b)
            for a, b in itertools.combinations(sorted(ap.index), 2)
            if ap.loc[a, "target"] == ap.loc[b, "target"]
        ]
    )
    np.testing.assert_array_equal(pairs.pos, expected)
    options = dict(null_size=31, null="joint", progress_bar=False)
    attached = get_map_pvalue(ap, ["target"], **options)
    explicit = get_map_pvalue(ap, ["target"], pos_pairs=expected, **options)
    pd.testing.assert_frame_equal(attached, explicit)


def test_component_null_against_independent_enumeration(label_example):
    """Match every placement of A against independently reconstructed labels and APs."""
    meta, feats, ap, pairs = label_example
    chosen = ap[ap.target == "a"]
    layouts, components, (ptr, neighbours) = permutation._label_layouts(
        [np.arange(2)],
        chosen.index.to_numpy(),
        chosen.n_pos_pairs.to_numpy(),
        (chosen.n_total_pairs - chosen.n_pos_pairs).to_numpy(),
        pairs,
    )
    group = layouts[0]
    pool = permutation._LabelPool(
        group.comps, components, ptr, neighbours, pairs.similarity
    )
    layout = permutation._LabelLayout(
        group.members, group.edges, group.lone, group.block_of, pool
    )
    placements = np.array(list(itertools.permutations(range(4), 2)))
    actual = permutation._label_maps_np(layout, pool.code[placements])
    unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    similarities = unit @ unit.T
    expected = []
    for receivers in placements:
        labels = [set(values) - {"a"} for values in meta.target]
        for receiver in receivers:
            labels[receiver].add("a")
        terms = []
        for receiver, positive in (receivers, receivers[::-1]):
            negatives = [
                j
                for j in range(4)
                if j != receiver and labels[receiver].isdisjoint(labels[j])
            ]
            before = sum(
                similarities[receiver, j] > similarities[receiver, positive]
                for j in negatives
            )
            terms.append(1 / (1 + before))
        expected.append(np.mean(terms))
    assert len(expected) == 12 and len(np.unique(expected)) > 1
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_scores_only_and_public_routing(label_example, hierarchical):
    """Route both public entry points to the fixed-draw multilabel engine."""
    _, _, ap, _ = label_example
    function = (
        mean_average_precision_hierarchical if hierarchical else mean_average_precision
    )
    options = dict(progress_bar=False)
    if hierarchical:
        options["hierarchical_by"] = ["family"]
    zero = function(ap, ["family", "target"], 0, 0.05, **options)
    assert zero.mean_average_precision.notna().all()
    for name in zero.columns:
        if "p_value" in name or "significant" in name or name.startswith("below_"):
            assert zero[name].isna().all()
    result = function(ap, ["family", "target"], 31, 0.05, **options)
    flat = mean_average_precision(
        ap, ["family", "target"], 31, 0.05, seed=0, progress_bar=False
    )
    np.testing.assert_array_equal(result.p_value, flat.p_value)


@pytest.mark.parametrize("draws", [0, 31])
def test_per_call_resources_and_reference_only_cap(label_example, draws, monkeypatch):
    """Exclude members from the cap and avoid similarity tables for zero draws."""
    _, _, ap, _ = label_example
    if draws == 0:

        def unexpected(*args, **kwargs):
            pytest.fail("scores-only call allocated similarity tables")

        monkeypatch.setattr(permutation._LabelPool, "__init__", unexpected)
    with pytest.raises(
        ValueError, match="2 reference profiles.*max_reference_profiles=1"
    ):
        get_map_pvalue(
            ap, ["target"], draws, max_reference_profiles=1, progress_bar=False
        )
    limited = get_map_pvalue(
        ap,
        ["target"],
        draws,
        max_reference_profiles=2,
        pool_memory_budget=1,
        key_table_memory_budget=0,
        max_workers=2,
        progress_bar=False,
    )
    default = get_map_pvalue(ap, ["target"], draws, progress_bar=False)
    pd.testing.assert_frame_equal(limited, default)
