"""Tests for the multilabel matching implementation."""

import numpy as np
import duckdb
import pandas as pd
import pytest

from tests.helpers import simulate_random_plates
from copairs.matching import find_pairs_multilabel

SEED = 42


def _label_set(labels):
    """Normalize a list for the naive matcher using DuckDB null semantics."""
    if labels is None or labels is pd.NA:
        return set()
    return {label for label in labels if not pd.isna(label)}


def get_naive_pairs(dframe: pd.DataFrame, sameby, diffby, multilabel_col: str):
    """Get pairs using a naive implementation."""
    dframe = dframe.copy()

    dframe[multilabel_col] = dframe[multilabel_col].apply(
        lambda labels: None if labels is None else _label_set(labels)
    )
    cross = dframe.reset_index().merge(
        dframe.reset_index(), how="cross", suffixes=("_x", "_y")
    )
    # Remove self/reversed pairs, matching the index ordering in the SQL matcher.
    cross = cross.query("index_x < index_y").copy()

    def all_diff(row):
        labels_x = row[f"{multilabel_col}_x"]
        labels_y = row[f"{multilabel_col}_y"]
        if labels_x is None:
            return False
        if labels_y is None:
            return True
        return len(labels_x & labels_y) == 0

    def any_equal(row):
        labels_x = row[f"{multilabel_col}_x"]
        labels_y = row[f"{multilabel_col}_y"]
        if labels_x is None or labels_y is None:
            return False
        return len(labels_x & labels_y) > 0

    cross[f"{multilabel_col}_all_diff"] = cross.apply(all_diff, axis=1)
    cross[f"{multilabel_col}_any_equal"] = cross.apply(any_equal, axis=1)
    mask = True
    for col in sameby:
        if col == multilabel_col:
            mask = cross[f"{col}_any_equal"] & mask
        else:
            mask = (cross[f"{col}_x"] == cross[f"{col}_y"]) & mask
    for col in diffby:
        if col == multilabel_col:
            mask = cross[f"{col}_all_diff"] & mask
        else:
            mask = (cross[f"{col}_x"] != cross[f"{col}_y"]) & mask

    pairs = cross.loc[mask, ["index_x", "index_y"]].drop_duplicates()
    return pairs.sort_values(["index_x", "index_y"]).reset_index(drop=True)


def check_naive(dframe, sameby, diffby, multilabel_col):
    """Check find_pairs_multilabel and naive generate the same pair details."""
    gt_pairs = get_naive_pairs(dframe, sameby, diffby, multilabel_col)
    result = find_pairs_multilabel(dframe, sameby, diffby, multilabel_col)

    if multilabel_col in sameby:
        pairs, keys, counts = result
        labels = np.repeat(keys, counts)
        actual_associations = [
            (label, int(index_x), int(index_y))
            for label, (index_x, index_y) in zip(labels, pairs)
        ]
        label_sets = dframe[multilabel_col].apply(_label_set)
        expected_associations = [
            (label, int(index_x), int(index_y))
            for index_x, index_y in gt_pairs.itertuples(index=False, name=None)
            for label in label_sets.loc[index_x] & label_sets.loc[index_y]
        ]
        expected_associations.sort(key=lambda item: (item[0], item[2], item[1]))

        assert actual_associations == expected_associations
        assert pairs.dtype == np.uint32
        assert counts.dtype == np.int64
    else:
        expected_pairs = gt_pairs.sort_values(["index_y", "index_x"]).itertuples(
            index=False, name=None
        )
        assert list(map(tuple, result)) == list(expected_pairs)
        assert result.dtype == np.uint32


def assert_matching_results_equal(actual, expected):
    """Assert equality for either multilabel matcher return shape."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_sameby():
    """Check the multilabel implementation with sameby."""
    multilabel_col = "c"
    sameby = ["c"]
    diffby = ["p", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_sameby_other_cols():
    """Check the multilabel implementation with sameby and other cols."""
    multilabel_col = "c"
    sameby = ["c", "p"]
    diffby = ["w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_diffby():
    """Check the multilabel implementation with sameby."""
    multilabel_col = "c"
    sameby = ["p"]
    diffby = ["c", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()

    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_diffby():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = []
    diffby = ["c"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_diffby_many_cols():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = []
    diffby = ["c", "w"]
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_only_sameby_many_cols():
    """Check the multilabel implementation with only diffby being equal to c."""
    multilabel_col = "c"
    sameby = ["c", "w"]
    diffby = []
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_accepts_tuples_inputs():
    """find_pairs_multilabel should accept tuples for sameby and diffby."""
    multilabel_col = "c"
    sameby = ("c",)
    diffby = ("p", "w")
    dframe = simulate_random_plates(
        n_compounds=4, n_replicates=5, plate_size=5, sameby=sameby, diffby=diffby
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()
    check_naive(dframe, sameby, diffby, multilabel_col)


def test_accepts_string_inputs():
    """find_pairs_multilabel should accept strings for sameby and diffby."""
    multilabel_col = "c"
    sameby = "c"
    diffby = "p"
    dframe = simulate_random_plates(
        n_compounds=4,
        n_replicates=5,
        plate_size=5,
        sameby=[sameby],
        diffby=[diffby],
    )
    dframe = dframe.groupby(["p", "w"])["c"].unique().reset_index()

    gt_pairs = get_naive_pairs(dframe, [sameby], [diffby], multilabel_col)
    with pytest.deprecated_call():
        vals = find_pairs_multilabel(dframe, sameby, diffby, multilabel_col)
    if multilabel_col == sameby:
        vals = vals[0]
    vals = pd.DataFrame(vals, columns=["index_x", "index_y"])
    vals = vals.sort_values(["index_x", "index_y"]).reset_index(drop=True)

    assert set(vals.apply(frozenset, axis=1)) == set(gt_pairs.apply(frozenset, axis=1))


@pytest.mark.parametrize(
    ("sameby", "diffby"),
    [
        (["labels"], []),
        (["labels", "cohort"], ["batch"]),
        (["labels"], ["cohort"]),
        (["cohort"], ["labels"]),
        ([], ["labels", "batch"]),
    ],
)
def test_repeated_null_and_empty_labels_match_naive(sameby, diffby):
    """Repeated and missing labels preserve pair and association semantics."""
    dframe = pd.DataFrame(
        {
            "cohort": ["g1", "g1", "g2", "g1", "g1", "g2"],
            "batch": range(6),
            "labels": [
                ["b", "a", "a"],
                ["a", "b"],
                ["b", None],
                [],
                None,
                [None, "a"],
            ],
        }
    )

    check_naive(dframe, sameby, diffby, "labels")


@pytest.mark.parametrize("multilabel_sameby", [True, False])
def test_quoted_identifiers_and_relation_match_pandas(multilabel_sameby):
    """Quoted columns and declared relations match the pandas paths."""
    multilabel_col = 'label "sets'
    dframe = pd.DataFrame(
        {
            "group name": ["g1", "g1", "g1", "g2"],
            "select": ["b1", "b2", "b3", "b4"],
            multilabel_col: [["a", "b"], ["a"], ["c"], ["a"]],
        }
    )
    if multilabel_sameby:
        sameby = [multilabel_col, "group name"]
        diffby = ["select"]
    else:
        sameby = ["group name"]
        diffby = [multilabel_col, "select"]

    check_naive(dframe, sameby, diffby, multilabel_col)
    pandas_result = find_pairs_multilabel(dframe, sameby, diffby, multilabel_col)
    relation = duckdb.from_df(dframe.reset_index())
    relation_result = find_pairs_multilabel(relation, sameby, diffby, multilabel_col)

    assert_matching_results_equal(relation_result, pandas_result)


def test_multilabel_association_dtypes_and_order():
    """Pairs stay grouped by sorted label and retain the established pair order."""
    dframe = pd.DataFrame(
        {
            "labels": [
                ["b", "a", "a"],
                ["a", "b"],
                ["b", None],
                [],
                None,
                [None, "a"],
            ]
        }
    )

    pairs, keys, counts = find_pairs_multilabel(dframe, ["labels"], [], "labels")

    np.testing.assert_array_equal(
        pairs,
        np.array(
            [[0, 1], [0, 5], [1, 5], [0, 1], [0, 2], [1, 2]],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(keys, np.array(["a", "b"], dtype=object))
    np.testing.assert_array_equal(counts, np.array([3, 3], dtype=np.int64))


def test_no_non_null_labels_returns_typed_empty_associations():
    """Null and empty lists do not create positive pair-label rows."""
    dframe = pd.DataFrame({"labels": [[], None, [None]]})

    pairs, keys, counts = find_pairs_multilabel(dframe, ["labels"], [], "labels")

    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.uint32
    assert keys.shape == (0,)
    assert counts.shape == (0,)
    assert counts.dtype == np.int64


@pytest.mark.parametrize(
    ("sameby", "diffby"),
    [
        (["labels", "cohort"], ["cohort"]),
        (["labels"], ["labels"]),
    ],
)
def test_sameby_diffby_must_remain_disjoint(sameby, diffby):
    """All constraints retain the disjoint-list validation from find_pairs."""
    dframe = pd.DataFrame({"cohort": ["g1", "g1"], "labels": [["a"], ["a"]]})

    with pytest.raises(ValueError, match="must be disjoint lists"):
        find_pairs_multilabel(dframe, sameby, diffby, "labels")
