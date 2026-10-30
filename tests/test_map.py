"""Tests for (mean) Average Precision calculation."""

import copy
import json
import pickle
import warnings
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from copairs import compute
from copairs.map import (
    get_map_pvalue,
    average_precision,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from tests.helpers import ap_frame, query_meta, simulate_random_dframe
from copairs.matching import UnpairedException, find_pairs
from copairs.map.multilabel import average_precision as multilabel_average_precision
from copairs.map.average_precision import attached_pairs

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


@pytest.mark.parametrize(
    "labels, plates, feats, no_neg_ix",
    [
        # Profile 1 sorts before profile 2, which has negatives.
        (
            ["A", "A", "B", "B"],
            [1, 2, 1, 1],
            [[1, 0], [0, 1], [1, 0.1], [0.9, 0.2]],
            1,
        ),
        # Profile 2 sorts after every profile that has negatives.
        (["A", "B", "A"], [1, 1, 2], [[1, 0], [0, 1], [0.6, 0.8]], 2),
    ],
)
def test_multilabel_profile_without_negatives(labels, plates, feats, no_neg_ix):
    """Test that a multilabel profile with positives but no negatives gets none."""
    meta = pd.DataFrame({"id": range(len(labels)), "l": labels, "plate": plates})
    feats = np.asarray(feats, dtype=float)
    args = (["l"], [], ["plate"], ["l"])

    expected = average_precision(meta, feats, *args, progress_bar=False)
    result = multilabel_average_precision(
        meta.assign(l=[[label] for label in labels]),
        feats,
        *args,
        "l",
        progress_bar=False,
    )
    expected = expected.set_index("id")
    result = result.set_index("id").sort_index()

    row = result.loc[no_neg_ix]
    assert row["average_precision"] == 1.0
    assert row["n_pos_pairs"] == 1
    assert row["n_total_pairs"] == 1

    cols = [
        "average_precision",
        "normalized_average_precision",
        "n_pos_pairs",
        "n_total_pairs",
    ]
    # The two functions store pair counts with different integer dtypes.
    pd.testing.assert_frame_equal(
        result[cols], expected.loc[result.index, cols], check_dtype=False
    )


def test_mean_average_precision_independent_of_other_groups(tmp_path):
    """A group's mAP p-value depends neither on the other groups nor on the cache."""
    ap_scores = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y"],
            "average_precision": [0.9, 0.8, 0.7, 0.6],
            "normalized_average_precision": [0.9, 0.8, 0.7, 0.6],
            "n_pos_pairs": [3, 3, 5, 5],
            "n_total_pairs": [20, 20, 30, 30],
        }
    )
    only_y = ap_scores[ap_scores["group"] == "Y"]
    kwargs = dict(
        sameby=["group"],
        null_size=200,
        threshold=0.05,
        seed=0,
        progress_bar=False,
        null="copied",  # the cache belongs to the copied null
    )
    shared = tmp_path / "shared"
    both = mean_average_precision(ap_scores, cache_dir=shared, **kwargs)
    cached = list(shared.rglob("*.npy"))
    after_both = mean_average_precision(only_y, cache_dir=shared, **kwargs)
    alone = mean_average_precision(only_y, cache_dir=tmp_path / "fresh", **kwargs)

    # The second run reads Y's null from the first run's cache.
    assert sorted(shared.rglob("*.npy")) == sorted(cached)
    pd.testing.assert_frame_equal(after_both, alone)
    y_in_both = both.loc[both["group"] == "Y", "p_value"].tolist()
    assert y_in_both == alone["p_value"].tolist()


def test_map_pvalue_replicates_use_the_joint_null():
    """Replicate groups require actual pairs, attached or explicitly supplied."""
    rng = np.random.default_rng(SEED)
    meta = query_meta()
    feats = rng.standard_normal((len(meta), 10))
    ap_scores, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    kwargs = dict(progress_bar=False, null="joint")
    bare = ap_scores.copy()
    bare.attrs.clear()
    with pytest.raises(ValueError, match="actual positive pairs"):
        get_map_pvalue(bare, ["g"], 20_000, SEED, **kwargs)
    attached = get_map_pvalue(ap_scores, ["g"], 20_000, SEED, **kwargs)
    given = get_map_pvalue(bare, ["g"], 20_000, SEED, pairs=pairs, **kwargs)
    explicit = get_map_pvalue(
        bare,
        ["g"],
        20_000,
        SEED,
        pos_pairs=find_pairs(meta, sameby=["g"], diffby=[]),
        **kwargs,
    )
    assert attached["p_value"].tolist() == explicit["p_value"].tolist()
    assert given["p_value"].tolist() == explicit["p_value"].tolist()
    assert ((attached["p_value"] > 0) & (attached["p_value"] <= 1)).all()


def test_map_pvalue_narrows_with_group_size():
    """The same mAP is more significant with more replicates."""
    rows = []
    for n in (2, 4, 8):
        rows += [(f"n{n}", 0.18, n - 1, n - 1 + 100)] * n
    frame = ap_frame(rows)
    result = get_map_pvalue(
        frame,
        ["g"],
        20_000,
        SEED,
        progress_bar=False,
        null="joint",
        pos_pairs=find_pairs(frame, sameby=["g"], diffby=[]),
    )
    p = result.set_index("g")["p_value"]
    assert p["n2"] > p["n4"] > p["n8"]


def test_map_pvalue_requires_explicit_copied_selection(tmp_path):
    """Missing graphs raise instead of silently selecting the copied model."""
    rows = [("odd", 0.3, 10, 50)] * 3 + [("rep", 0.3, 2, 52)] * 3
    with pytest.raises(ValueError, match="actual positive pairs"):
        get_map_pvalue(
            ap_frame(rows),
            ["g"],
            1000,
            SEED,
            progress_bar=False,
            cache_dir=tmp_path,
            null="joint",
        )
    result = get_map_pvalue(
        ap_frame(rows),
        ["g"],
        1000,
        SEED,
        progress_bar=False,
        cache_dir=tmp_path,
        null="copied",
    )
    p = result.set_index("g")["p_value"]
    assert 0 < p["odd"] <= 1 and 0 < p["rep"] <= 1


def test_map_pvalue_one_profile_groups_use_their_ap_null():
    """A group of one profile, positives in other groups, gets that AP's null."""
    genes = ["a"] * 4 + ["b"] * 4 + ["neg"] * 10  # one profile per guide
    meta = pd.DataFrame({"gene": genes, "guide": range(len(genes))})
    meta["ref"] = np.where(meta["gene"] == "neg", meta.index, -1)
    feats = np.random.default_rng(SEED).standard_normal((len(meta), 8))
    ap = average_precision(
        meta, feats, ["gene", "ref"], [], [], ["gene", "ref"], progress_bar=False
    )
    ap = ap[ap["gene"] != "neg"]
    kwargs = dict(progress_bar=False, null="joint")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inferred = get_map_pvalue(ap, ["guide"], 20_000, SEED, **kwargs)
    pos_pairs = find_pairs(meta, sameby=["gene", "ref"], diffby=[])
    explicit = get_map_pvalue(
        ap, ["guide"], 20_000, SEED, pos_pairs=pos_pairs, **kwargs
    )
    assert inferred["p_value"].tolist() == explicit["p_value"].tolist()
    ranks = itertools.combinations(range(13), 3)  # 3 sibling guides, 10 controls
    null = np.array([np.mean(np.arange(1, 4) / (np.array(r) + 1)) for r in ranks])
    exact = [(null >= m - 1e-9).mean() for m in inferred["mean_average_precision"]]
    np.testing.assert_allclose(inferred["p_value"], exact, atol=0.01)


def test_joint_null_reads_attached_pairs():
    """The joint null takes edges from attached pairs, so a reset or repeated index raises."""
    rng = np.random.default_rng(SEED)
    meta = query_meta(others=3)
    meta = meta.sample(frac=1, random_state=1).reset_index(drop=True)
    feats = rng.standard_normal((len(meta), 10))
    ap = average_precision(meta, feats, ["g"], [], [], ["g"], progress_bar=False)
    kept = ap[ap["g"] != "other"]
    bare = kept.copy()
    pairs = attached_pairs(bare)
    bare.attrs.clear()
    kwargs = dict(progress_bar=False, null="joint")
    got = get_map_pvalue(kept, ["g"], 2000, SEED, **kwargs)
    expected = get_map_pvalue(bare, ["g"], 2000, SEED, pairs=pairs, **kwargs)
    assert got["p_value"].tolist() == expected["p_value"].tolist()
    reset = kept.reset_index(drop=True)  # pairs now misaligned
    for null in ("joint", "permutation"):
        with pytest.raises(ValueError, match="no reset_index"):
            get_map_pvalue(reset, ["g"], 2000, SEED, progress_bar=False, null=null)
    doubled = pd.concat([kept, kept])  # keeps the pairs, repeats every row
    for null in ("joint", "permutation"):
        with pytest.raises(ValueError, match="repeats row numbers"):
            get_map_pvalue(doubled, ["g"], 2000, SEED, progress_bar=False, null=null)


def test_joint_null_uses_attached_pairs_for_outside_positives():
    """Groups whose positives lie in other groups get edges from the attached pairs."""
    rows = [("T", "A")] * 3 + [("T", "B")] * 2 + [("ctrl", "ctrl")] * 10
    meta = pd.DataFrame(rows, columns=["t", "comp"])
    meta["ctrl"] = meta["t"] == "ctrl"
    feats = np.random.default_rng(SEED).standard_normal((len(meta), 8))
    ap, pairs = average_precision(
        meta, feats, ["t"], ["comp"], [], ["ctrl"], progress_bar=False,
        return_pairs=True,
    )  # fmt: skip
    ap = ap[~ap["ctrl"]]  # A's positives are B's two members, and B's are A's
    assert ap["n_pos_pairs"].tolist() == [2, 2, 2, 3, 3]
    kwargs = dict(progress_bar=False, null="joint")
    explicit = get_map_pvalue(
        ap, ["comp"], 5000, SEED, pos_pairs=pairs["pos"], **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback to the copied null
        attached = get_map_pvalue(ap, ["comp"], 5000, SEED, **kwargs)
    assert attached["p_value"].tolist() == explicit["p_value"].tolist()


@pytest.mark.parametrize("null", ["permutation", "joint"])
def test_pairs_need_the_ap_frames_row_index(null):
    """A relabelled AP index raises a clear error wherever pairs index it."""
    meta = query_meta()
    feats = np.random.default_rng(SEED).standard_normal((len(meta), 10))
    ap = average_precision(meta, feats, ["g"], [], [], ["g"], progress_bar=False)
    for index in (meta["g"] + meta.index.astype(str), -1 - meta.index):
        relabelled = ap.set_axis(index)
        with pytest.raises(ValueError, match="index must hold the row numbers"):
            get_map_pvalue(relabelled, ["g"], 100, SEED, progress_bar=False, null=null)
    bare = ap.set_axis(meta["g"] + meta.index.astype(str))
    bare.attrs.clear()
    with pytest.raises(ValueError, match="actual positive pairs"):
        get_map_pvalue(bare, ["g"], 100, SEED, progress_bar=False, null="joint")
    assert len(
        get_map_pvalue(bare, ["g"], 100, SEED, progress_bar=False, null="copied")
    )


def test_ap_frames_carry_their_pairs():
    """AP frames carry their pairs through copies and filters, not saving or merging."""
    rng = np.random.default_rng(SEED)
    meta = pd.DataFrame({"g": ["q"] * 3 + [f"c{i}" for i in range(6)]})
    feats = rng.standard_normal((len(meta), 5))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    assert type(pairs) is dict
    assert set(pairs) == {"pos", "neg", "similarity"}
    assert set(ap) == {
        *meta.columns,
        "average_precision",
        "normalized_average_precision",
        "n_pos_pairs",
        "n_total_pairs",
    }
    assert attached_pairs(ap) is pairs
    assert attached_pairs(ap[ap["g"] == "q"].copy()) is pairs
    assert attached_pairs(copy.deepcopy(ap)) is pairs
    assert attached_pairs(pd.concat([ap.iloc[:4], ap.iloc[4:]])) is pairs
    assert json.loads(json.dumps(ap.attrs)) == {"copairs_pairs": {}}  # as to_parquet
    assert attached_pairs(pickle.loads(pickle.dumps(ap))) is None
    other = average_precision(meta, feats, ["g"], [], [], ["g"], progress_bar=False)
    assert attached_pairs(pd.concat([ap, other])) is None
    # The pairs index the frame's rows and rank with the AP stage's similarities.
    ap_again = average_precision(
        meta.set_axis(range(10, 19)), feats, ["g"], [], [], ["g"], progress_bar=False
    )
    np.testing.assert_array_equal(attached_pairs(ap_again)["pos"], pairs["pos"])
    similarity = compute.get_similarity_fn("cosine", progress_bar=False)
    np.testing.assert_array_equal(
        pairs["similarity"](pairs["neg"]), similarity(feats, pairs["neg"], 20000)
    )


@pytest.mark.parametrize("distance", ["cosine", "correlation", "cityblock"])
def test_pair_similarity_runs_in_the_calling_thread(distance, monkeypatch):
    """The pairs' similarity starts no thread pool and matches the AP stage's."""
    meta = query_meta(others=5)
    feats = np.random.default_rng(SEED).standard_normal((len(meta), 6))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], batch_size=7, distance=distance,
        progress_bar=False, return_pairs=True,
    )  # fmt: skip
    similarity = compute.get_similarity_fn(distance, progress_bar=False)
    expected = similarity(feats, pairs["neg"], 7)
    ap = ap[ap["g"] != "other"]
    before = mean_average_precision(ap, ["g"], 200, 0.05, SEED, progress_bar=False)

    def no_pool(*args, **kwargs):
        raise AssertionError("parallel_map called")

    monkeypatch.setattr(compute, "parallel_map", no_pool)
    got = pairs["similarity"](pairs["neg"])
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    after = mean_average_precision(ap, ["g"], 200, 0.05, SEED, progress_bar=False)
    assert after["p_value"].tolist() == before["p_value"].tolist()


def test_map_pvalue_rejects_misaligned_pos_pairs():
    """pos_pairs from a meta whose index has gaps no longer index ap_scores."""
    rng = np.random.default_rng(SEED)
    meta = query_meta()
    meta = meta.drop(index=[1, 7, 20])  # filtered by the user: index 0, 2, 3, 4, ...
    feats = rng.standard_normal((len(meta), 10))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    kwargs = dict(progress_bar=False, null="joint")
    inferred = get_map_pvalue(ap, ["g"], 2000, SEED, **kwargs)
    explicit = get_map_pvalue(ap, ["g"], 2000, SEED, pairs=pairs, **kwargs)
    assert inferred["p_value"].tolist() == explicit["p_value"].tolist()
    with pytest.raises(ValueError, match="must index the rows"):
        get_map_pvalue(
            ap,
            ["g"],
            2000,
            SEED,
            pos_pairs=find_pairs(meta, sameby=["g"], diffby=[]),
            **kwargs,
        )


def run_map_entry(ap_scores, hierarchical, **kwargs):
    """Run the mAP of group "g" from either entry point, hierarchical over "g"."""
    kwargs = dict(null_size=100, threshold=0.05, seed=SEED, progress_bar=False) | kwargs
    if hierarchical:
        return mean_average_precision_hierarchical(
            ap_scores.assign(h=0), ["g", "h"], hierarchical_by=["g"], **kwargs
        )
    return mean_average_precision(ap_scores, ["g"], **kwargs)


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("null", ["permutation", "joint", "copied"])
def test_no_valid_ap_scores_raise_under_every_null(null, hierarchical, tmp_path):
    """AP frames with no valid AP score or group key raise, with their pairs attached."""
    meta = query_meta()
    feats = np.random.default_rng(SEED).standard_normal((len(meta), 10))
    ap = average_precision(meta, feats, ["g"], [], [], ["g"], progress_bar=False)
    for frame in (ap.iloc[:0], ap.assign(average_precision=np.nan), ap.assign(g=None)):
        with pytest.raises(UnpairedException, match="valid AP scores"):
            run_map_entry(frame, hierarchical, null=null, cache_dir=tmp_path)


def test_indices_keep_rows_without_sameby_values(tmp_path):
    """`indices` count rows missing a sameby value; only invalid APs are dropped."""
    ap = ap_frame(
        [
            (None, 0.5, 1, 10),
            ("a", np.nan, 1, 10),
            ("a", 0.4, 1, 10),
            ("b", 0.3, 0, 10),
            ("b", 0.2, 1, 10),
        ]
    )
    result = mean_average_precision(
        ap,
        ["g"],
        100,
        0.05,
        SEED,
        progress_bar=False,
        null="copied",
        cache_dir=tmp_path,
    )
    assert result.set_index("g")["indices"].to_dict() == {"a": [1], "b": [2]}


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(null="exact"), "unknown null"),
        (dict(null_size=-1), "null_size"),
        (dict(seed=-1), "seed must be"),
        (dict(seed=-1, null="copied"), "seed must be"),
        (dict(pos_pairs=np.zeros((0, 2), dtype=int)), "joint null only"),
        (dict(pairs=None, null="permutation"), "needs the pairs"),
    ],
)
def test_empty_input_checks_arguments_first(bad, match, hierarchical):
    """Bad arguments raise their own error on AP frames with no valid AP score."""
    empty = ap_frame([("q", 0.5, 1, 10)]).iloc[:0]
    empty.attrs.clear()  # no attached pairs
    with pytest.raises(ValueError, match=match):
        run_map_entry(empty, hierarchical, **bad)
