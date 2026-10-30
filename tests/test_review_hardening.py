"""Regressions for the four findings from the adjustments re-review."""

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from copairs.map import (
    get_map_pvalue,
    average_precision,
    mean_average_precision_hierarchical,
)
from copairs.map.average_precision import attached_pairs
from tests.test_review_adjustments import profiles


@pytest.mark.parametrize("draws", [0, 20])
@pytest.mark.parametrize("reload", [False, True])
def test_raw_pairs_reject_mixed_sources(draws, reload):
    """Disjoint row indices and equal degrees do not authenticate mixed AP calls."""
    meta = pd.DataFrame(
        {
            "g": ["a", "b"] * 3 + [f"c{i}" for i in range(6)],
            "h": ["x"] * 3 + ["y"] * 3 + [f"c{i}" for i in range(6)],
        }
    )
    feats = np.random.default_rng(18).normal(size=(12, 5))
    a, pa = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    b = average_precision(meta, feats, ["h"], [], [], ["h"], progress_bar=False)
    mixed = pd.concat([a[a.g == "a"], b[b.g == "b"]])
    assert attached_pairs(mixed) is None
    if reload:
        mixed = pickle.loads(pickle.dumps(mixed))
    with pytest.raises(ValueError, match="sources|identit"):
        get_map_pvalue(
            mixed, ["g"], draws, null="joint", pos_pairs=pa.pos, progress_bar=False
        )


@pytest.mark.parametrize("reload", [False, True])
def test_same_source_raw_pairs_survive_lost_attachment(reload):
    """Same-source raw joint pairs remain supported after a copy or results-only save."""
    ap, pairs = profiles()
    expected = get_map_pvalue(ap, ["g"], 20, null="joint", progress_bar=False)
    frame = pd.concat([ap.iloc[:5], ap.iloc[5:]])
    frame = pickle.loads(pickle.dumps(frame)) if reload else frame.copy()
    frame.attrs.clear()
    got = get_map_pvalue(
        frame, ["g"], 20, null="joint", pos_pairs=pairs.pos, progress_bar=False
    )
    pd.testing.assert_frame_equal(got, expected)


def disconnected():
    """Return q with two disconnected components, another group and controls."""
    meta = pd.DataFrame(
        {
            "g": ["q"] * 4 + ["r"] * 2 + ["c0", "c1"],
            "plate": ["A", "A", "B", "B", "C", "C", "C", "C"],
        }
    )
    feats = np.random.default_rng(13).normal(size=(8, 5))
    return average_precision(
        meta,
        feats,
        ["g", "plate"],
        [],
        [],
        ["g"],
        progress_bar=False,
        return_pairs=True,
    )


@pytest.mark.parametrize("draws", [0, 20])
@pytest.mark.parametrize("missing", ["dropped", "AP", "key", "positives"])
def test_disconnected_partial_group_rejected(draws, missing):
    """Losing an entire graph component still removes members of the tested group."""
    ap, _ = disconnected()
    if missing == "dropped":
        ap = ap.drop(index=[2, 3])
    else:
        column, value = {
            "AP": ("average_precision", np.nan),
            "key": ("g", None),
            "positives": ("n_pos_pairs", 0),
        }[missing]
        ap.loc[[2, 3], column] = value
    with pytest.raises(ValueError, match="incomplete|partial|recompute"):
        get_map_pvalue(ap, ["g"], draws, progress_bar=False)


def test_complete_groups_follow_requested_keys():
    """Dropping whole requested groups remains valid, including disconnected components."""
    ap, pairs = disconnected()
    kwargs = dict(null_size=20, progress_bar=False)
    only_r = get_map_pvalue(ap[ap.g == "r"], ["g"], **kwargs)
    both = get_map_pvalue(ap, ["g"], **kwargs)
    assert only_r.p_value.tolist() == both.loc[both.g == "r", "p_value"].tolist()
    # With g+plate as the requested grouping, each component is a whole group.
    part = get_map_pvalue(ap.drop(index=[2, 3]), ["g", "plate"], **kwargs)
    assert len(part) == 2
    assert attached_pairs(ap.copy()).source_metadata is pairs.source_metadata


def test_changed_group_keys_cannot_hide_incomplete_groups():
    """Renaming a surviving component cannot bypass original-membership validation."""
    ap, _ = disconnected()
    frame = ap.drop(index=[2, 3]).copy()
    frame.loc[frame.g == "q", "g"] = "renamed"
    with pytest.raises(ValueError, match="group|metadata"):
        get_map_pvalue(frame, ["g"], 0, progress_bar=False)
    # The retained source snapshot must not change with edits of the output frame.
    assert attached_pairs(ap).source_metadata.loc[0, "g"] == "q"


def test_new_group_columns_need_all_original_contributors():
    """New grouping columns can be validated on a full AP table, not an unknown subset."""
    ap, _ = disconnected()
    full = ap.assign(extra="all")
    assert len(get_map_pvalue(full, ["g", "extra"], 0, progress_bar=False)) == 2
    with pytest.raises(ValueError, match="group|metadata"):
        get_map_pvalue(full.drop(index=[2, 3]), ["g", "extra"], 0, progress_bar=False)


@pytest.mark.parametrize("draws", [0, 20])
@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.parametrize("dtype", ["Int64", "UInt32", "int64", "uint32", "float64"])
def test_numeric_count_dtypes(dtype, null, draws, tmp_path):
    """All models normalize valid integer-valued count columns consistently."""
    ap, _ = profiles()
    converted = ap.astype({"n_pos_pairs": dtype, "n_total_pairs": dtype})
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        get_map_pvalue(converted, ["g"], draws, **kwargs),
        get_map_pvalue(ap, ["g"], draws, **kwargs),
    )


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
def test_arrow_count_dtypes(null, tmp_path):
    """Arrow-backed integral columns follow the same validation and sampling path."""
    pytest.importorskip("pyarrow")
    ap, _ = profiles()
    converted = ap.astype(
        {"n_pos_pairs": "int64[pyarrow]", "n_total_pairs": "uint32[pyarrow]"}
    )
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        get_map_pvalue(converted, ["g"], 20, **kwargs),
        get_map_pvalue(ap, ["g"], 20, **kwargs),
    )


@pytest.mark.parametrize("draws", [0, 10])
@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
def test_workers_validated_before_sampling(draws, workers, null):
    """Invalid worker settings raise even without an executor or for joint sampling."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="max_workers"):
        get_map_pvalue(
            ap, ["g"], draws, null=null, max_workers=workers, progress_bar=False
        )


@pytest.mark.parametrize("draws", [0, 10])
@pytest.mark.parametrize("keys", [[], ["g", "g"], ["absent"], [None]])
def test_group_columns_validated_before_sampling(draws, keys):
    """Empty, duplicate, missing and invalid column names get useful errors."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="sameby|column"):
        get_map_pvalue(ap, keys, draws, progress_bar=False)


@pytest.mark.parametrize("draws", [0, 10])
@pytest.mark.parametrize("keys", [[], ["g", "g"], [None]])
def test_hierarchy_columns_validated_before_sampling(draws, keys):
    """Scores-only cannot bypass invalid hierarchy specifications."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="hierarchical_by|column"):
        mean_average_precision_hierarchical(
            ap, ["batch", "g"], draws, 0.05, hierarchical_by=keys, progress_bar=False
        )


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
def test_count_validation_preserves_large_integers(dtype):
    """Validation does not round integer counts through float64."""
    from copairs.map.map import _normalize_pair_counts

    counts = pd.DataFrame(
        {"n_pos_pairs": [2**53 + 1], "n_total_pairs": [2**63 - 1]}
    ).astype(dtype)
    np.testing.assert_array_equal(
        _normalize_pair_counts(counts), [[2**53 + 1, 2**63 - 1]]
    )


@pytest.mark.parametrize(
    "value,dtype",
    [
        (np.nan, "float64"),
        (np.inf, "float64"),
        (1.5, "float64"),
        (-1, "int64"),
        (2**63, "uint64"),
        (2**63, "float64"),
        (True, "bool"),
        (pd.NA, "Int64"),
    ],
)
def test_invalid_counts_raise_useful_errors(value, dtype):
    """Missing, fractional, nonfinite and out-of-range counts never reach a sampler."""
    from copairs.map.map import _normalize_pair_counts

    counts = pd.DataFrame(
        {"n_pos_pairs": [1], "n_total_pairs": pd.Series([value], dtype=dtype)}
    )
    with pytest.raises(ValueError, match="pair counts"):
        _normalize_pair_counts(counts)


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_excluded_group_counts_are_not_cast(null, tmp_path):
    """Excluded missing-key rows can contain missing counts without contaminating other groups."""
    ap, _ = profiles()
    expected = get_map_pvalue(
        ap[ap.g == "b"], ["g"], 20, null=null, progress_bar=False, cache_dir=tmp_path
    )
    ap["n_total_pairs"] = ap.n_total_pairs.astype(float)
    ap.loc[ap.g == "a", ["g", "n_total_pairs"]] = [None, np.nan]
    got = get_map_pvalue(
        ap, ["g"], 20, null=null, progress_bar=False, cache_dir=tmp_path
    )
    assert got.p_value.tolist() == expected.p_value.tolist()


@pytest.mark.parametrize("format", ["csv", "parquet", "pickle"])
def test_saved_results_can_use_original_handle(format, tmp_path):
    """Results-only saves preserve row tokens; the explicit original handle restores validation."""
    ap, pairs = disconnected()
    path = tmp_path / f"scores.{format}"
    if format == "csv":
        ap.to_csv(path)
        loaded = pd.read_csv(path, index_col=0)
    elif format == "parquet":
        pytest.importorskip("pyarrow")
        ap.to_parquet(path)
        loaded = pd.read_parquet(path)
    else:
        ap.to_pickle(path)
        loaded = pd.read_pickle(path)
    assert attached_pairs(loaded) is None
    expected = get_map_pvalue(ap, ["g"], 20, progress_bar=False)
    got = get_map_pvalue(loaded, ["g"], 20, pairs=pairs, progress_bar=False)
    np.testing.assert_allclose(
        got.mean_average_precision, expected.mean_average_precision
    )
    assert got.p_value.tolist() == expected.p_value.tolist()
    with pytest.raises(ValueError, match="incomplete"):
        get_map_pvalue(
            loaded.drop(index=[2, 3]), ["g"], 0, pairs=pairs, progress_bar=False
        )


def test_source_metadata_isolated_from_output_edits():
    """Original grouping membership does not change when the returned frame is edited."""
    ap, pairs = disconnected()
    ap.loc[ap.g == "q", "g"] = "changed"
    assert pairs.source_metadata.loc[0, "g"] == "q"


def test_permutation_requires_original_group_metadata():
    """An explicit handle cannot silently drop the source membership contract."""
    ap, pairs = disconnected()
    incomplete = replace(pairs, source_metadata=None)
    with pytest.raises(ValueError, match="metadata|membership"):
        get_map_pvalue(ap, ["g"], 0, pairs=incomplete, progress_bar=False)
