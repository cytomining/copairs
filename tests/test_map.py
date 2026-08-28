"""Tests for (mean) Average Precision calculation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from copairs import compute
from copairs.map import average_precision
from tests.helpers import simulate_random_dframe
from copairs.matching import UnpairedException
from copairs.map.multilabel import average_precision as multilabel_average_precision

SEED = 0


def _with_layout(feats: np.ndarray, layout: str) -> np.ndarray:
    """Return feature values with the requested memory layout."""
    if layout == "fortran":
        return np.asfortranarray(feats)
    if layout == "strided":
        backing = np.empty((len(feats), feats.shape[1] * 2), dtype=feats.dtype)
        backing[:, ::2] = feats
        return backing[:, ::2]
    return feats


def binary2indices(arr: np.ndarray) -> np.ndarray:
    """Convert a binary matrix to a list of indices."""
    return np.nonzero(arr)[1].reshape(arr.shape[0], -1)


def test_random_binary_matrix():
    """Test the random binary matrix generation."""
    rng = np.random.default_rng(SEED)

    # Test with n=3, m=4, k=2
    indices = compute.random_binary_matrix(3, 4, 2, rng)
    assert indices.shape == (3, 2)
    assert np.all(indices < 4)
    assert np.all(indices >= 0)
    assert np.unique(indices, axis=1).shape == indices.shape

    # Test with n=5, m=6, k=3
    indices = compute.random_binary_matrix(5, 6, 3, rng)
    assert indices.shape == (5, 3)
    assert np.all(indices < 6)
    assert np.all(indices >= 0)
    assert np.unique(indices, axis=1).shape == indices.shape


def test_compute_ap():
    """Test the average precision computation."""
    num_pos, num_neg, num_perm = 5, 6, 100
    total = num_pos + num_neg

    y_true = np.zeros((num_perm, total), dtype=int)
    y_true[:, :num_pos] = 1
    y_pred = np.random.uniform(0, 1, [num_perm, total])
    df = pd.DataFrame(
        {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }
    )
    rel_k = (
        df["y_pred"]
        .apply(lambda x: np.argsort(x)[::-1])
        .apply(lambda x: np.array(df.y_true[0])[x])
    )
    rel_k = np.stack(rel_k)

    ap = compute.average_precision(binary2indices(rel_k))

    ap_sklearn = df.apply(
        lambda x: average_precision_score(x["y_true"], x["y_pred"]), axis=1
    )

    assert np.allclose(ap_sklearn, ap)


def test_compute_ap_contiguous():
    """Test the contiguous average precision computation."""
    num_pos_range = [2, 9]
    num_neg_range = [10, 20]
    num_samples_range = [5, 30]
    rng = np.random.default_rng(SEED)
    for _ in range(30):
        num_samples = rng.integers(*num_samples_range)
        counts, rel_k_list = [], []
        ground_truth = []
        null_confs_gt = np.empty((num_samples, 2), dtype=int)
        for j in range(num_samples):
            num_pos = rng.integers(*num_pos_range)
            num_neg = rng.integers(*num_neg_range)
            total = num_pos + num_neg
            y_true = np.zeros(total, dtype=int)
            y_true[:num_pos] = 1
            y_pred = np.random.uniform(0, 1, total)
            ap_score = average_precision_score(y_true, y_pred)
            ground_truth.append(ap_score)

            rel_k = y_true[np.argsort(y_pred)[::-1]]
            rel_k_list.append(rel_k)
            counts.append(total)
            null_confs_gt[j] = [num_pos, total]

        rel_k_list = np.concatenate(rel_k_list)
        counts = np.asarray(counts)
        ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
        assert np.allclose(null_confs_gt, null_confs)
        assert np.allclose(ap_scores, ground_truth)


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pipeline(progress_bar: bool):
    """Check the implementation with for mAP calculation."""
    length = 10
    vocab_size = {"p": 5, "w": 3, "l": 4}
    n_feats = 5
    pos_sameby = ["l"]
    pos_diffby = ["p"]
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))
    average_precision(
        meta,
        feats,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        progress_bar=progress_bar,
    )


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pipeline_multilabel(progress_bar: bool):
    """Check the multilabel implementation with for mAP calculation."""
    length = 10
    vocab_size = {"p": 3, "w": 5, "l": 4}
    n_feats = 8
    multilabel_col = "l"
    pos_sameby = ["l"]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    meta = meta.groupby(["p", "w"])["l"].unique().reset_index()
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))

    multilabel_average_precision(
        meta,
        feats,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        multilabel_col,
        progress_bar=progress_bar,
    )


@pytest.mark.parametrize("progress_bar", [True, False])
def test_raise_no_pairs(progress_bar: bool):
    """Test the exception raised when no pairs are found."""
    length = 10
    vocab_size = {"p": 3, "w": 3, "l": 10}
    n_feats = 5
    pos_sameby = ["l"]
    pos_diffby = ["p"]
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    meta.drop_duplicates(subset=pos_sameby, inplace=True)
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))
    with pytest.raises(UnpairedException, match="Unable to find positive pairs."):
        average_precision(
            meta,
            feats,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            progress_bar=progress_bar,
        )
    with pytest.raises(UnpairedException, match="Unable to find negative pairs."):
        average_precision(
            meta, feats, pos_diffby, [], pos_sameby, [], progress_bar=progress_bar
        )


def test_raise_nan_error():
    """Test the exception raised when there are null values."""
    length = 10
    vocab_size = {"p": 5, "w": 3, "l": 4}
    n_feats = 8
    pos_sameby = ["l"]
    pos_diffby = ["p"]
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))

    # add null values
    feats_nan = feats.copy()
    feats_nan[2, 2] = None
    meta_nan = meta.copy()
    meta_nan.loc[1, "p"] = None

    with pytest.raises(ValueError, match="features should not have null values."):
        average_precision(
            meta, feats_nan, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )
    with pytest.raises(
        ValueError, match="metadata columns should not have null values."
    ):
        average_precision(
            meta_nan, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_cosine_fast_path_exact_regular_parity_with_ties(dtype, layout):
    """The cosine fast path preserves positive-first ranking for exact ties."""
    meta = pd.DataFrame(
        {
            "compound": ["a", "a", "b", "b", "c", "c"],
            "plate": ["p1", "p2", "p1", "p2", "p1", "p2"],
        }
    )
    # Every pair has exactly the same cosine similarity, so the expected AP of
    # one verifies that positive pairs still precede tied negative pairs.
    feats = np.tile(
        np.asarray([0.125, -1.5, 2.25, 0.75, -0.0625], dtype=dtype),
        (len(meta), 1),
    )
    feats = _with_layout(feats, layout)
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["compound"],
        "progress_bar": False,
        "batch_size": 2,
    }

    generic = average_precision(distance=compute.pairwise_cosine, **kwargs)
    optimized = average_precision(distance="cosine", **kwargs)

    pd.testing.assert_frame_equal(optimized, generic, check_exact=True)
    np.testing.assert_array_equal(optimized["average_precision"], 1.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_cosine_fast_path_exact_multilabel_parity_with_ties(dtype, layout):
    """Multilabel cosine retrieval preserves exact grouped tie semantics."""
    meta = pd.DataFrame(
        {
            "compound": ["a", "b", "c", "d"],
            "target": [["x"], ["x", "y"], ["y"], ["z"]],
        }
    )
    feats = np.tile(
        np.asarray([0.125, -1.5, 2.25, 0.75, -0.0625], dtype=dtype),
        (len(meta), 1),
    )
    feats = _with_layout(feats, layout)
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["target"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["target"],
        "multilabel_col": "target",
        "progress_bar": False,
        "batch_size": 2,
    }

    generic = multilabel_average_precision(
        distance=compute.pairwise_cosine, **kwargs
    )
    optimized = multilabel_average_precision(distance="cosine", **kwargs)

    pd.testing.assert_frame_equal(optimized, generic, check_exact=True)
    np.testing.assert_array_equal(optimized["average_precision"], 1.0)


def test_cosine_fast_path_regular_skips_unpaired_nonfinite_rows():
    """Regular cosine preparation only normalizes profiles found in pairs."""
    meta = pd.DataFrame(
        {
            "compound": ["a", "a", "b", "b", "zero", "inf"],
            "cohort": ["main", "main", "main", "main", "zero", "inf"],
        }
    )
    feats = np.asarray(
        [
            [1.0, 0.5],
            [0.75, 1.0],
            [-0.5, 1.0],
            [1.0, -0.25],
            [0.0, 0.0],
            [np.inf, 1.0],
        ]
    )
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": ["cohort"],
        "neg_diffby": ["compound"],
        "progress_bar": False,
        "batch_size": 2,
    }

    with np.errstate(all="raise"):
        generic = average_precision(distance=compute.pairwise_cosine, **kwargs)
        optimized = average_precision(distance="cosine", **kwargs)

    pd.testing.assert_frame_equal(optimized, generic, check_exact=True)


def test_cosine_fast_path_multilabel_skips_unpaired_nonfinite_rows():
    """Multilabel preparation only normalizes profiles found in pairs."""
    meta = pd.DataFrame(
        {
            "target": [["x"], ["x"], ["y"], ["y"], ["zero"], ["inf"]],
            "cohort": ["main", "main", "main", "main", "zero", "inf"],
        }
    )
    feats = np.asarray(
        [
            [1.0, 0.5],
            [0.75, 1.0],
            [-0.5, 1.0],
            [1.0, -0.25],
            [0.0, 0.0],
            [np.inf, 1.0],
        ]
    )
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["target"],
        "pos_diffby": [],
        "neg_sameby": ["cohort"],
        "neg_diffby": ["target"],
        "multilabel_col": "target",
        "progress_bar": False,
        "batch_size": 2,
    }

    with np.errstate(all="raise"):
        generic = multilabel_average_precision(
            distance=compute.pairwise_cosine, **kwargs
        )
        optimized = multilabel_average_precision(distance="cosine", **kwargs)

    pd.testing.assert_frame_equal(optimized, generic, check_exact=True)


def test_generic_string_and_callable_similarity_fallback(monkeypatch):
    """Non-cosine strings and custom callables retain generic batch processing."""
    meta = pd.DataFrame(
        {"compound": ["a", "a", "b", "b"], "plate": ["p1", "p2", "p1", "p2"]}
    )
    feats = np.eye(len(meta))
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["compound"],
        "progress_bar": False,
    }

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("cosine fast path must not handle fallback metrics")

    monkeypatch.setattr(compute, "cosine_pairs", fail_fast_path)
    string_result = average_precision(distance="euclidean", **kwargs)

    calls = []

    def custom_distance(x_sample, y_sample):
        calls.append(len(x_sample))
        return compute.pairwise_euclidean(x_sample, y_sample)

    callable_result = average_precision(distance=custom_distance, **kwargs)

    pd.testing.assert_frame_equal(string_result, callable_result, check_exact=True)
    assert calls


def test_cosine_fast_path_forwards_progress_setting(monkeypatch):
    """Each cosine pair set retains the requested progress-bar setting."""
    progress_settings = []

    def sequential_map(par_func, items, progress_bar=True):
        progress_settings.append(progress_bar)
        for item in items:
            par_func(item)

    monkeypatch.setattr(compute, "parallel_map", sequential_map)
    meta = pd.DataFrame({"compound": ["a", "a", "b", "b"]})
    average_precision(
        meta,
        np.eye(len(meta)),
        pos_sameby=["compound"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=["compound"],
        progress_bar=False,
    )

    assert progress_settings == [False, False]


def test_progress_bar_consistency():
    """Test that the progress_bar argument does not change results."""
    length = 10
    vocab_size = {"p": 5, "w": 3, "l": 4}
    n_feats = 5
    pos_sameby = ["l"]
    pos_diffby = ["p"]
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))
    with_pb, no_pb = [
        average_precision(
            meta,
            feats,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            progress_bar=progress_bar,
        )
        for progress_bar in (True, False)
    ]
    assert with_pb.equals(no_pb), "The progress_bar argument changed results"


def test_multilabel_has_normalized_ap():
    """Test that multilabel AP includes normalized_average_precision column."""
    length = 10
    vocab_size = {"p": 3, "w": 5, "l": 4}
    n_feats = 8
    multilabel_col = "l"
    rng = np.random.default_rng(SEED)

    meta = simulate_random_dframe(length, vocab_size, ["l"], [], rng)
    meta = meta.groupby(["p", "w"])["l"].unique().reset_index()
    feats = rng.uniform(size=(len(meta), n_feats))

    result = multilabel_average_precision(
        meta, feats, ["l"], [], [], ["l"], multilabel_col
    )

    assert "normalized_average_precision" in result.columns
