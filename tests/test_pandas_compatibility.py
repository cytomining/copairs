"""Regression tests for pandas string and multilabel representations."""

import numpy as np
import pandas as pd
import pytest

from copairs.matching import find_pairs, find_pairs_multilabel


def _check_multilabel_pairs(dframe, shared_label):
    """Check pairs, label counts, and input preservation for three profiles."""
    original = dframe.copy(deep=True)
    pairs, keys, counts = find_pairs_multilabel(dframe, ["labels"], [], "labels")
    np.testing.assert_array_equal(pairs, [[0, 1]])
    np.testing.assert_array_equal(keys, [shared_label])
    np.testing.assert_array_equal(counts, [1])

    pairs = find_pairs_multilabel(dframe, [], ["labels"], "labels")
    assert set(map(tuple, pairs)) == {(0, 2), (1, 2)}
    pd.testing.assert_frame_equal(dframe, original)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_inferred_string_columns(storage):
    """DuckDB must accept pandas 3's inferred str dtype with either backend."""
    if storage == "pyarrow":
        pytest.importorskip("pyarrow")
    with pd.option_context("mode.string_storage", storage):
        dframe = pd.DataFrame({"label": ["a", "a", "b"]})
    np.testing.assert_array_equal(find_pairs(dframe, ["label"], []), [[0, 1]])
    pairs = find_pairs(dframe, [], ["label"])
    assert set(map(tuple, pairs)) == {(0, 2), (1, 2)}


@pytest.mark.parametrize("dtype", [None, "object", "string[python]", "string[pyarrow]"])
def test_groupby_unique_string_labels(dtype):
    """Normalize StringArray cells produced by groupby.unique, not just ndarrays."""
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    dframe = pd.DataFrame(
        {
            "sample": [0, 0, 1, 2],
            "labels": pd.Series(["a", "b", "a", "c"], dtype=dtype),
        }
    )
    dframe = dframe.groupby("sample")["labels"].unique().reset_index(drop=True)
    _check_multilabel_pairs(dframe.to_frame(), "a")


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "bool"])
def test_numpy_label_arrays(dtype):
    """Do not turn numeric arrays into lists of unsupported NumPy scalars."""
    dframe = pd.DataFrame(
        {"labels": [np.array([value], dtype=dtype) for value in [1, 1, 0]]}
    )
    _check_multilabel_pairs(dframe, 1)


@pytest.mark.parametrize("dtype", ["Int64", "Float32", "boolean"])
def test_nullable_label_arrays(dtype):
    """Convert extension arrays to native lists while preserving inner nulls."""
    dframe = pd.DataFrame(
        {
            "labels": [
                pd.array([1, pd.NA], dtype=dtype),
                pd.array([1], dtype=dtype),
                pd.array([0], dtype=dtype),
            ]
        }
    )
    _check_multilabel_pairs(dframe, 1)


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(None, id="none"),
        pytest.param(np.nan, id="nan"),
        pytest.param(pd.NA, id="pd-na"),
        pytest.param([], id="empty-list"),
    ],
)
def test_missing_or_empty_label_cells(missing):
    """Preserve existing SQL behavior for missing and empty multilabel cells."""
    dframe = pd.DataFrame({"labels": [["a"], ["a"], missing]})
    _check_multilabel_pairs(dframe, "a")
