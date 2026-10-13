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


def _ap_frame(rows):
    """Build an `ap_scores` frame from (group, ap, n_pos_pairs, n_total_pairs) rows."""
    frame = pd.DataFrame(
        rows, columns=["g", "average_precision", "n_pos_pairs", "n_total_pairs"]
    )
    frame["normalized_average_precision"] = 0.0
    return frame


def test_map_pvalue_replicates_use_the_joint_null():
    """Replicate groups get the joint null, inferred or from explicit `pos_pairs`."""
    from copairs.map import get_map_pvalue
    from copairs.matching import find_pairs

    rng = np.random.default_rng(SEED)
    meta = pd.DataFrame({"g": ["query"] * 4 + [f"control{i}" for i in range(30)]})
    feats = rng.standard_normal((len(meta), 10))
    ap_scores = average_precision(meta, feats, ["g"], [], [], ["g"], progress_bar=False)
    kwargs = dict(progress_bar=False, null="joint")
    bare = ap_scores.copy()
    bare.attrs.clear()  # no attached pairs: positives are inferred
    inferred = get_map_pvalue(bare, ["g"], 20_000, SEED, **kwargs)
    attached = get_map_pvalue(ap_scores, ["g"], 20_000, SEED, **kwargs)
    explicit = get_map_pvalue(
        bare,
        ["g"],
        20_000,
        SEED,
        pos_pairs=find_pairs(meta, sameby=["g"], diffby=[]),
        **kwargs,
    )
    assert attached["p_value"].tolist() == explicit["p_value"].tolist()
    assert inferred["p_value"].tolist() == explicit["p_value"].tolist()
    assert ((inferred["p_value"] > 0) & (inferred["p_value"] <= 1)).all()


def test_map_pvalue_narrows_with_group_size():
    """The same mAP is more significant with more replicates."""
    from copairs.map import get_map_pvalue

    rows = []
    for n in (2, 4, 8):
        rows += [(f"n{n}", 0.18, n - 1, n - 1 + 100)] * n
    result = get_map_pvalue(
        _ap_frame(rows), ["g"], 20_000, SEED, progress_bar=False, null="joint"
    )
    p = result.set_index("g")["p_value"]
    assert p["n2"] > p["n4"] > p["n8"]


def test_map_pvalue_falls_back_with_warning(tmp_path):
    """Groups whose positives are not their other members use the previous null."""
    from copairs.map import get_map_pvalue

    rows = [("odd", 0.3, 10, 50)] * 3 + [("rep", 0.3, 2, 52)] * 3
    with pytest.warns(UserWarning, match="pos_pairs"):
        result = get_map_pvalue(
            _ap_frame(rows),
            ["g"],
            1000,
            SEED,
            progress_bar=False,
            cache_dir=tmp_path,
            null="joint",
        )
    p = result.set_index("g")["p_value"]
    assert 0 < p["odd"] <= 1 and 0 < p["rep"] <= 1


def test_map_pvalue_rejects_misaligned_pos_pairs():
    """pos_pairs from a meta whose index has gaps no longer index ap_scores."""
    from copairs.map import get_map_pvalue
    from copairs.matching import find_pairs

    rng = np.random.default_rng(SEED)
    meta = pd.DataFrame({"g": ["query"] * 4 + [f"control{i}" for i in range(30)]})
    meta = meta.drop(index=[1, 7, 20])  # filtered by the user: index 0, 2, 3, 4, ...
    feats = rng.standard_normal((len(meta), 10))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    bare = ap.copy()
    bare.attrs.clear()
    kwargs = dict(progress_bar=False, null="joint")
    inferred = get_map_pvalue(bare, ["g"], 2000, SEED, **kwargs)
    explicit = get_map_pvalue(bare, ["g"], 2000, SEED, pairs=pairs, **kwargs)
    assert inferred["p_value"].tolist() == explicit["p_value"].tolist()
    with pytest.raises(ValueError, match="pos_pairs must index"):
        get_map_pvalue(
            ap,
            ["g"],
            2000,
            SEED,
            progress_bar=False,
            pos_pairs=find_pairs(meta, sameby=["g"], diffby=[]),
            null="joint",
        )
