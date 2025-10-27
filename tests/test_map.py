"""Tests for (mean) Average Precision calculation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

from copairs import compute
from copairs.map import average_precision
from tests.helpers import simulate_random_dframe
from copairs.matching import UnpairedException
from copairs.map.multilabel import average_precision as multilabel_average_precision

SEED = 0


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


def test_compute_auc_contiguous():
    """Test the contiguous roc_auc computation."""
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
            roc_score = roc_auc_score(y_true, y_pred)
            ground_truth.append(roc_score)

            rel_k = y_true[np.argsort(y_pred)[::-1]]
            rel_k_list.append(rel_k)
            counts.append(total)
            null_confs_gt[j] = [num_pos, total]

        rel_k_list = np.concatenate(rel_k_list)
        counts = np.asarray(counts)
        auc_scores, null_confs = compute.auc_contiguous(rel_k_list, counts)
        assert np.allclose(null_confs_gt, null_confs)
        assert np.allclose(auc_scores, ground_truth)


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
