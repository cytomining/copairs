"""Tests for inverted-index multilabel matching and per-label AP (Numba)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from copairs import fastap, compute, matching
from copairs.map import multilabel

N_LABELS = 12


def label_frame(seed=0, n=120, empty=True):
    """Rows with 0-3 labels each and two monolabel columns."""
    rng = np.random.default_rng(seed)
    labels = [
        list(
            rng.choice(
                N_LABELS, size=rng.integers(0 if empty else 1, 4), replace=False
            ).astype(str)
        )
        for _ in range(n)
    ]
    return pd.DataFrame(
        {
            "labels": labels,
            "plate": rng.integers(3, size=n).astype(str),
            "well": rng.integers(5, size=n).astype(str),
        }
    )


def sql_pairs(*args):
    """find_pairs_multilabel through the DuckDB implementation."""
    return matching.find_pairs_multilabel(*args, backend="numpy")


def index_pairs(*args):
    """find_pairs_multilabel through the inverted label index, where supported."""
    return matching.find_pairs_multilabel(*args, backend="numba")


def indexed(dframe):
    """Whether the inverted label index can represent ``dframe``'s labels."""
    return fastap.multilabel.multilabel_pairs(dframe, "labels") is not None


def as_set(pairs):
    """Unordered pair set."""
    return set(map(tuple, np.sort(np.asarray(pairs, dtype=np.int64), axis=1)))


@pytest.mark.parametrize(
    "sameby,diffby",
    [
        (["labels"], []),
        (["labels", "plate"], []),
        (["labels"], ["well"]),
        (["labels"], ["labels"]),  # share a label, but not the whole list
    ],
)
def test_shared_label_pairs_match_sql(sameby, diffby):
    """Pairs sharing a label, and per-label counts, equal the SQL result."""
    dframe = label_frame()
    args = (dframe, sameby, diffby, "labels")
    assert indexed(dframe)
    pairs, keys, counts = index_pairs(*args)
    sql_pairs_, sql_keys, sql_counts = sql_pairs(*args)
    np.testing.assert_array_equal(keys, sql_keys)
    np.testing.assert_array_equal(counts, sql_counts)
    start = 0
    for count in counts:  # same pairs label by label
        got = as_set(pairs[start : start + count])
        expected = as_set(sql_pairs_[start : start + count])
        assert got == expected
        start += count
    assert pairs.dtype == np.uint32


@pytest.mark.parametrize(
    "sameby,diffby",
    [([], ["labels"]), (["plate"], ["labels"]), ([], ["labels", "well"])],
)
def test_disjoint_label_pairs_match_sql(sameby, diffby):
    """Pairs sharing no label equal the SQL result, sorted and unique."""
    dframe = label_frame(1)
    args = (dframe, sameby, diffby, "labels")
    assert indexed(dframe)
    pairs = index_pairs(*args)
    assert as_set(pairs) == as_set(sql_pairs(*args))
    keys = pairs[:, 0].astype(np.int64) * len(dframe) + pairs[:, 1]
    assert (np.diff(keys) > 0).all() and (pairs[:, 0] < pairs[:, 1]).all()


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    "sameby,diffby",
    [
        (["labels", "plate"], ["plate"]),
        (["labels", "plate"], ["labels", "plate"]),
        (["plate"], ["labels", "plate"]),
    ],
)
def test_overlapping_sameby_diffby_raise(backend, sameby, diffby):
    """A monolabel column in both sameby and diffby raises, as in find_pairs."""
    with pytest.raises(ValueError, match="sameby and diffby must be disjoint lists"):
        matching.find_pairs_multilabel(
            label_frame(), sameby, diffby, "labels", backend=backend
        )


@pytest.mark.parametrize(
    "plate",
    [
        [None, "p0", "p1"],  # missing values never match in SQL
        [1, "1", 2.0],  # mixed Python types, converted by DuckDB
        [np.nan, 0.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    "sameby,diffby", [(["labels", "plate"], []), (["labels"], ["plate"])]
)
def test_shared_label_pairs_with_irregular_columns(plate, sameby, diffby):
    """Columns with missing or mixed values filter label pairs as in SQL."""
    dframe = label_frame(4, empty=False)
    dframe["plate"] = pd.Series(plate * (len(dframe) // 3), dtype=object)
    args = (dframe, sameby, diffby, "labels")
    got = index_pairs(*args)
    expected = sql_pairs(*args)
    np.testing.assert_array_equal(got[1], expected[1])
    np.testing.assert_array_equal(got[2], expected[2])
    assert as_set(got[0]) == as_set(expected[0])


def test_array_label_cells_with_other_columns():
    """NumPy array cells work with other columns; only those reach DuckDB."""
    dframe = label_frame(3)
    arrays = dframe.assign(labels=dframe["labels"].map(np.array))
    assert indexed(arrays)
    args = (["labels"], ["plate"], "labels")
    for g, e in zip(index_pairs(arrays, *args), index_pairs(dframe, *args)):
        np.testing.assert_array_equal(g, e)
    args = (["plate"], ["labels"], "labels")
    np.testing.assert_array_equal(
        index_pairs(arrays, *args), index_pairs(dframe, *args)
    )


@pytest.mark.parametrize(
    "labels,index",
    [
        ([["a", None], ["a"], ["b"]], None),  # missing value inside a list
        ([["a", "a"], ["a"], ["b"]], None),  # duplicate label in a row
        ([["a"], [1], ["b"]], None),  # mixed label types
        ([["a"], ["a"], ["b"]], [5, 6, 7]),  # non-default index
    ],
)
def test_unsupported_labels_fall_back(labels, index):
    """Inputs the inverted index cannot represent exactly use the SQL path."""
    dframe = pd.DataFrame({"labels": labels}, index=index)
    assert not indexed(dframe)


def consistency_input(seed=2):
    """Profiles with labels such that every profile has negatives."""
    dframe = label_frame(seed, n=150, empty=False)
    feats = np.random.default_rng(seed).normal(size=(len(dframe), 10))
    return dframe, feats


CONSISTENCY = dict(
    pos_sameby=["labels"], pos_diffby=[], neg_sameby=[], neg_diffby=["labels"]
)


@pytest.mark.parametrize("int_labels", [False, True])
def test_multilabel_ap_matches_numpy(int_labels):
    """Per-label APs and label column equal the NumPy loop's."""
    dframe, feats = consistency_input()
    if int_labels:
        dframe["labels"] = dframe["labels"].map(lambda v: [int(x) for x in v])

    kwargs = dict(
        CONSISTENCY,
        multilabel_col="labels",
        distance=compute.pairwise_cosine,
        progress_bar=False,
    )
    fast = multilabel.average_precision(dframe, feats, backend="numba", **kwargs)
    numpy = multilabel.average_precision(dframe, feats, backend="numpy", **kwargs)
    pd.testing.assert_frame_equal(
        fast.drop(columns=["average_precision", "normalized_average_precision"]),
        numpy.drop(columns=["average_precision", "normalized_average_precision"]),
    )
    for col in ["average_precision", "normalized_average_precision"]:
        np.testing.assert_allclose(fast[col], numpy[col], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "big,array_dtype", [(2, None), (2**31 - 1, None), (2**31, None), (2, np.int32)]
)
def test_numeric_label_keys(big, array_dtype):
    """Integer labels give keys of the SQL path's values and dtype.

    DuckDB stores Python ints as int32 while they fit, and keeps the dtype of
    NumPy label arrays.
    """
    labels = [[1, big], [big], [1, 3], [3]]
    if array_dtype is not None:
        labels = [np.array(v, dtype=array_dtype) for v in labels]
    dframe = pd.DataFrame({"labels": labels})
    _, keys, counts = index_pairs(dframe, ["labels"], [], "labels")
    _, sql_keys, sql_counts = sql_pairs(dframe, ["labels"], [], "labels")
    assert keys.dtype == np.asarray(sql_keys).dtype
    np.testing.assert_array_equal(keys, sql_keys)
    np.testing.assert_array_equal(counts, sql_counts)


@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
def test_multilabel_ap_with_sql_fallback(distance):
    """Labels the index cannot represent match through SQL, with the same APs."""
    dframe, feats = consistency_input(5)
    dframe.loc[0, "labels"] = dframe.loc[0, "labels"] * 2  # a label twice in a row
    assert not indexed(dframe)
    kwargs = dict(
        CONSISTENCY,
        neg_sameby=["plate"],
        multilabel_col="labels",
        distance=distance,
        progress_bar=False,
    )
    fast = multilabel.average_precision(dframe, feats, backend="numba", **kwargs)
    numpy = multilabel.average_precision(dframe, feats, backend="numpy", **kwargs)
    ap_columns = ["average_precision", "normalized_average_precision"]
    pd.testing.assert_frame_equal(
        fast.drop(columns=ap_columns), numpy.drop(columns=ap_columns)
    )
    for col in ap_columns:
        np.testing.assert_allclose(fast[col], numpy[col], rtol=1e-6, atol=1e-6)


def test_multilabel_ap_builds_label_index_once(monkeypatch):
    """Positive and negative pairs come from one label index per call."""
    calls = []
    build = fastap.multilabel.multilabel_pairs

    def counted(*args):
        calls.append(args)
        return build(*args)

    monkeypatch.setattr(fastap.multilabel, "multilabel_pairs", counted)
    dframe, feats = consistency_input()
    multilabel.average_precision(
        dframe,
        feats,
        **CONSISTENCY,
        multilabel_col="labels",
        progress_bar=False,
        backend="numba",
    )
    assert len(calls) == 1
