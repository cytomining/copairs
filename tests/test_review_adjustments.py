"""Regression tests for the agreed 0.6.0 mAP review policies."""

import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from copairs.map import (
    get_map_pvalue,
    average_precision,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from copairs.nulls import joint, permutation
from copairs.map.hierarchical_fdr import (
    apply_fdr_correction,
    apply_hierarchical_fdr_correction,
)


def profiles():
    """Return interleaved groups and controls with equal within-group degrees."""
    meta = pd.DataFrame({"g": ["a", "b"] * 3 + [f"c{i}" for i in range(6)]})
    meta["batch"] = "one"
    feats = np.random.default_rng(3).normal(size=(len(meta), 5))
    return average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )


@pytest.mark.parametrize("null", ["joint", "permutation"])
@pytest.mark.parametrize("equal_scores", [False, True])
def test_equal_degree_reset_rejected(null, equal_scores):
    """A reset cannot silently reuse identities even when scores/counts match."""
    ap, pairs = profiles()
    if equal_scores:
        ap["average_precision"] = 0.5
    bad = ap[ap.g == "b"].reset_index(drop=True)
    with pytest.raises(ValueError, match="row identit|reset_index"):
        get_map_pvalue(bad, ["g"], 10, null=null, pairs=pairs, progress_bar=False)


def test_joint_missing_pairs_never_falls_back(monkeypatch):
    """A multi-member group without its graph raises before any copied sampling."""
    ap, _ = profiles()
    ap.attrs.clear()
    module = importlib.import_module("copairs.map.map")

    def forbidden(*args, **kwargs):
        raise AssertionError("copied fallback called")

    monkeypatch.setattr(module, "_copied_pvalues", forbidden)
    with pytest.raises(ValueError, match="positive pairs|pos_pairs"):
        get_map_pvalue(ap, ["g"], 10, null="joint", progress_bar=False)


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_zero_draws_are_untested(null, hierarchical, monkeypatch, tmp_path):
    """Zero draws keep scores but do not sample or run a correction."""
    ap, _ = profiles()
    module = importlib.import_module("copairs.map.map")

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling or correction called")

    for name in (
        "_copied_pvalues",
        "apply_fdr_correction",
        "apply_hierarchical_fdr_correction",
    ):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(joint, "joint_pvalues", forbidden)
    monkeypatch.setattr(permutation, "_Pool", forbidden)
    fn = mean_average_precision_hierarchical if hierarchical else mean_average_precision
    extra = {"hierarchical_by": ["batch"]} if hierarchical else {}
    out = fn(
        ap,
        ["batch", "g"],
        0,
        0.05,
        null=null,
        progress_bar=False,
        cache_dir=tmp_path,
        **extra,
    )
    assert out.mean_average_precision.notna().all()
    numeric = ["p_value", "corrected_p_value"]
    flags = ["below_p", "below_corrected_p"]
    if hierarchical:
        numeric += ["stage1_p_value", "stage1_corrected_p_value"]
        flags += ["stage1_significant"]
    assert out[numeric].isna().all().all()
    assert out[flags].isna().all().all()
    for column in flags:
        assert str(out[column].dtype) == "boolean"


def test_default_seed_and_explicit_pairs_precedence():
    """Omitted seed equals zero and explicit valid pairs override attached state."""
    ap, pairs = profiles()
    other, _ = profiles()
    from copairs.map.average_precision import attach_pairs

    attach_pairs(ap, other.attrs["copairs_pairs"].pairs)
    kwargs = dict(progress_bar=False, pairs=pairs, max_workers=1)
    a = mean_average_precision(ap, ["g"], 40, 0.05, **kwargs)
    b = mean_average_precision(ap, ["g"], 40, 0.05, seed=0, **kwargs)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="row identit"):
        mean_average_precision(ap, ["g"], 40, 0.05, progress_bar=False)


def test_explicit_positive_pairs_must_agree_with_provenance():
    """Raw positive arrays cannot bypass the attached row-identity checks."""
    ap, pairs = profiles()
    bad = ap[ap.g == "b"].reset_index(drop=True)
    with pytest.raises(ValueError, match="row identit|reset_index"):
        get_map_pvalue(
            bad, ["g"], 10, null="joint", pos_pairs=pairs.pos, progress_bar=False
        )
    with pytest.raises(ValueError, match="positive pairs|pos_pairs"):
        get_map_pvalue(
            ap,
            ["g"],
            10,
            null="joint",
            pos_pairs=np.array([[0, 1]]),
            progress_bar=False,
        )


def test_matched_plate_restrictions_have_identical_results():
    """A positive-only metadata guard must not reject an otherwise identical graph."""
    meta = pd.DataFrame(
        {
            "g": ["q", "q", "c0", "c1", "c2", "c3"],
            "ctrl": [False, False, True, True, True, True],
            "plate": [0, 1, 0, 0, 1, 1],
        }
    )
    feats = np.random.default_rng(0).normal(size=(6, 4))
    results = []
    for diffby in ([], ["plate"]):
        ap = average_precision(
            meta, feats, ["g"], diffby, [], ["ctrl", "plate"], progress_bar=False
        )
        results.append(mean_average_precision(ap, ["g"], 100, 0.05, progress_bar=False))
    pd.testing.assert_frame_equal(*results)


def test_memory_arguments_are_per_call_and_result_invariant():
    """Calls can use different budgets concurrently without changing globals/results."""
    ap, _ = profiles()
    limits = dict(
        max_reference_profiles=9, pool_memory_budget=1, key_table_memory_budget=0
    )
    before = (
        permutation.MAX_POOL,
        permutation.POOL_BUDGET,
        permutation.KEY_TABLE_BUDGET,
    )
    expected = mean_average_precision(
        ap, ["g"], 30, 0.05, progress_bar=False, max_workers=1
    )
    with ThreadPoolExecutor(2) as executor:
        futures = [
            executor.submit(
                mean_average_precision,
                ap,
                ["g"],
                30,
                0.05,
                progress_bar=False,
                max_workers=w,
                **settings,
            )
            for w, settings in ((1, limits), (2, {}))
        ]
        for future in futures:
            pd.testing.assert_frame_equal(expected, future.result(timeout=30))
    assert before == (
        permutation.MAX_POOL,
        permutation.POOL_BUDGET,
        permutation.KEY_TABLE_BUDGET,
    )
    with pytest.raises(ValueError, match="9.*8"):
        mean_average_precision(
            ap, ["g"], 30, 0.05, max_reference_profiles=8, progress_bar=False
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_reference_profiles", -1),
        ("max_reference_profiles", 1.5),
        ("pool_memory_budget", 0),
        ("pool_memory_budget", True),
        ("key_table_memory_budget", -1),
        ("key_table_memory_budget", float("inf")),
    ],
)
def test_resource_validation(name, value):
    """Resource settings reject invalid values before allocating tables."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match=name):
        get_map_pvalue(ap, ["g"], 0, progress_bar=False, **{name: value})


def test_zero_draws_still_validate_incomplete_groups():
    """Scores-only does not turn an unsupported requested layout into success."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="dropped|no longer have"):
        get_map_pvalue(ap.drop(index=0), ["g"], 0, progress_bar=False)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_correction_keeps_untested_rows_missing(hierarchical):
    """Missing tests do not contaminate valid corrected results or become p=1."""
    frame = pd.DataFrame(
        {"g": ["a", "a", "b"], "dose": [0, 1, 0], "p_value": [0.01, np.nan, np.nan]}
    )
    if hierarchical:
        out = apply_hierarchical_fdr_correction(frame, ["g"], ["g", "dose"])
        assert pd.isna(out.loc[2, "stage1_significant"])
    else:
        out = apply_fdr_correction(frame)
    assert out.loc[0, "corrected_p_value"] == 0.01
    assert out.loc[[1, 2], "corrected_p_value"].isna().all()


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
def test_none_seed_resolves_entropy_once(null, monkeypatch, tmp_path):
    """Each run resolves one seed before dispatch; None never fills the copied cache."""
    ap, _ = profiles()
    original = np.random.SeedSequence
    fresh = []

    def sequence(*args, **kwargs):
        if not args and not kwargs:
            fresh.append(True)
            return original(17)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.random, "SeedSequence", sequence)
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path, max_workers=2)
    got = get_map_pvalue(ap, ["g"], 30, seed=None, **kwargs)
    assert len(fresh) == 1
    assert not list(tmp_path.rglob("*.npy"))
    expected = get_map_pvalue(ap, ["g"], 30, seed=17, **kwargs)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
def test_public_entry_points_default_seed_zero(null, tmp_path):
    """All public mAP entry points have the same explicit/implicit seed semantics."""
    ap, _ = profiles()
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path)
    for fn, args, extra in (
        (get_map_pvalue, (ap, ["g"], 20), {}),
        (mean_average_precision, (ap, ["g"], 20, 0.05), {}),
        (
            mean_average_precision_hierarchical,
            (ap, ["batch", "g"], 20, 0.05),
            {"hierarchical_by": ["batch"]},
        ),
    ):
        pd.testing.assert_frame_equal(
            fn(*args, **extra, **kwargs), fn(*args, seed=0, **extra, **kwargs)
        )
        zero_args = list(args)
        zero_args[2] = 0
        assert fn(*zero_args, **extra, **kwargs).p_value.isna().all()


@pytest.mark.parametrize("null_size", [0, 10])
@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.parametrize(
    "column,value",
    [("average_precision", np.inf), ("n_total_pairs", 1), ("n_pos_pairs", 1.5)],
)
def test_scores_only_still_rejects_invalid_data(null_size, null, column, value):
    """Missing tests are not a fallback for invalid scores or pair counts."""
    ap, _ = profiles()
    ap[column] = value
    with pytest.raises(ValueError, match="counts must|scores must"):
        get_map_pvalue(ap, ["g"], null_size, null=null, progress_bar=False)


def test_pair_validation_and_backing_arrays():
    """Canonical arrays stay shared; malformed, duplicate or self pairs raise."""
    from copairs.nulls.common import check_pairs

    canonical = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    assert check_pairs(canonical) is canonical
    np.testing.assert_array_equal(check_pairs(canonical[::-1, ::-1]), canonical)
    for invalid in (
        np.array([[0, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[-1, 2]]),
        canonical.astype(float),
    ):
        with pytest.raises(ValueError):
            check_pairs(invalid)


def test_missing_rows_are_reported(caplog):
    """Whole excluded groups keep other groups valid and exclusions are visible."""
    ap, _ = profiles()
    ap.loc[ap.g == "a", "g"] = None
    got = get_map_pvalue(ap, ["g"], 0, progress_bar=False)
    assert got.g.tolist() == ["b"]
    assert "6 rows with missing AP/no positive pairs" in caplog.text
    assert "3 rows with missing group keys" in caplog.text


def test_noncomplete_positive_graph_against_independent_enumeration():
    """Every assignment preserves the path graph, even across positive plate rules."""
    import itertools

    meta = pd.DataFrame(
        {"g": ["q", "q", "q", "c0", "c1"], "plate": ["A", "A", "B", "A", "B"]}
    )
    feats = np.random.default_rng(8).normal(size=(5, 4))
    ap, pairs = average_precision(
        meta,
        feats,
        ["g"],
        ["plate"],
        [],
        ["g"],
        return_pairs=True,
        progress_bar=False,
    )
    kept = ap[ap.g == "q"]
    members, edges, blocks, block_of = permutation.group_layouts(
        [np.arange(3)],
        kept.index.to_numpy(),
        kept.n_pos_pairs.to_numpy(),
        (kept.n_total_pairs - kept.n_pos_pairs).to_numpy(),
        pairs.pos,
        pairs.neg,
    )[0]
    np.testing.assert_array_equal(edges, [[0, 2], [1, 2]])
    pool = permutation._Pool(blocks, pairs.similarity)
    layout = permutation._Layout(members, edges, block_of, pairs.similarity, pool)
    assignments = np.array(list(itertools.permutations(range(5), 3)))
    expected = []
    for chosen in assignments:
        negatives = sorted(set(range(5)) - set(chosen))
        scores = []
        for slot in range(3):
            neighbors = [j if i == slot else i for i, j in edges if slot in (i, j)]
            targets = [chosen[j] for j in neighbors] + negatives
            sims = pairs.similarity(
                np.column_stack([np.full(len(targets), chosen[slot]), targets])
            )
            relevance = np.arange(len(targets)) < len(neighbors)
            order = np.argsort(1 - sims, kind="stable")
            ranks = np.flatnonzero(relevance[order]) + 1
            scores.append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        expected.append(np.mean(scores))
    np.testing.assert_allclose(
        permutation._group_maps_np(layout, assignments), expected, atol=1e-14, rtol=0
    )
    # Exercise the public path that formerly rejected this positive-only rule.
    got = get_map_pvalue(ap, ["g"], 200, progress_bar=False)
    assert len(got) == 1 and 0 < got.p_value.iloc[0] <= 1


def test_hierarchical_resource_propagation():
    """Hierarchical dispatch honors the same pre-allocation cap, including zero draws."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="9.*8"):
        mean_average_precision_hierarchical(
            ap,
            ["batch", "g"],
            0,
            0.05,
            hierarchical_by=["batch"],
            max_reference_profiles=8,
            progress_bar=False,
        )
