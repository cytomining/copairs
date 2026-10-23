"""Tests for the Numba similarity kernels and counting-based average precision."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from copairs import fastap, compute
from copairs.map import average_precision
from tests.helpers import simulate_random_dframe
from copairs.map.average_precision import build_rank_lists


def random_pairs(rng, n_profiles, n_pos, n_neg):
    """Random positive and negative pairs over ``n_profiles`` profiles."""
    pos = rng.integers(n_profiles, size=(n_pos, 2))
    neg = rng.integers(n_profiles, size=(n_neg, 2))
    return pos, neg


def rank_list_ap(pos, neg, pos_sims, neg_sims):
    """AP from copairs' rank lists."""
    paired_ix, rel_k_list, counts = build_rank_lists(pos, neg, pos_sims, neg_sims)
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    return paired_ix, ap_scores, null_confs


@pytest.mark.parametrize("ties", [False, True])
def test_ap_from_pairs_matches_rank_lists(ties):
    """Counting AP equals rank-list AP, including tied similarities."""
    rng = np.random.default_rng(0)
    pos, neg = random_pairs(rng, 300, 900, 20_000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    if ties:
        pos_sims = np.round(pos_sims * 8) / 8
        neg_sims = np.round(neg_sims * 8) / 8
    expected = rank_list_ap(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[2], expected[2])
    assert got[2].dtype == expected[2].dtype
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


def test_ap_from_pairs_edge_cases():
    """Profiles without positives get NaN; NaN similarities rank last."""
    rng = np.random.default_rng(1)
    pos, neg = random_pairs(rng, 50, 40, 500)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    pos_sims[::7] = np.nan
    neg_sims[::5] = np.nan
    expected = rank_list_ap(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(np.isnan(got[1]), np.isnan(expected[1]))
    assert np.isnan(got[1]).any()
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


def test_ap_from_pairs_histogram_chunks():
    """Splitting negatives over per-thread histograms leaves APs unchanged."""
    rng = np.random.default_rng(7)
    pos, neg = random_pairs(rng, 100, 300, 5000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    whole = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims, budget_bytes=1)
    split = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    for w, s in zip(whole, split):
        np.testing.assert_array_equal(w, s)


def test_ap_from_pairs_accepts_series_similarities():
    """Pandas similarities are converted to arrays."""
    pos, neg = np.array([[0, 1]]), np.array([[0, 2], [1, 2]])
    pos_sims, neg_sims = np.array([0.5]), np.array([0.1, 0.9])
    expected = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pd.Series(pos_sims), pd.Series(neg_sims))
    np.testing.assert_array_equal(got[1], expected[1])


METRICS = list(fastap.FAST_METRICS)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pair_similarity_matches_generic(metric, dtype):
    """Kernel similarities match copairs' generic functions to float32 rounding."""
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(200, 37)).astype(dtype)
    pairs = rng.integers(200, size=(3000, 2))
    generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 512)
    got = fastap.pair_similarity(feats, metric)(pairs)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, generic, rtol=1e-5, atol=1e-6)


def test_pair_similarity_without_kernel():
    """Metrics without a kernel and callables fall back to the generic path."""
    feats = np.ones((3, 2))
    assert fastap.pair_similarity(feats, "jaccard") is None
    assert fastap.pair_similarity(feats, lambda x, y: x[:, 0]) is None
    generic = compute.get_similarity_fn("jaccard", progress_bar=False)
    assert fastap.similarity_fn(feats, "jaccard", generic) is generic


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_pair_similarity_nonfinite_features(metric):
    """Inf - inf differences give NaN similarities, as in the generic functions."""
    feats = np.array([[np.inf, 1.0, 0.0], [np.inf, 0.0, 0.0], [1.0, 2.0, 3.0]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    with np.errstate(invalid="ignore"):
        generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 8)
    got = fastap.pair_similarity(feats, metric)(pairs)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(generic))
    np.testing.assert_allclose(got, generic, rtol=1e-6)


def test_pair_similarity_float32_overflow():
    """Rows whose float32 norm overflows get their cosine, not the generic 0."""
    feats = np.array(
        [[1e20, 2e20, 0.0], [2e20, 1e20, 0.0], [1.0, 2.0, 0.0]], np.float32
    )
    pairs = np.array([[0, 1], [0, 2]])
    got = fastap.pair_similarity(feats, "cosine")(pairs)
    np.testing.assert_allclose(got, [0.8, 1.0], rtol=1e-6)


def simulated_pipeline_input(seed=0):
    """Metadata and features with replicated labels."""
    rng = np.random.default_rng(seed)
    vocab_size = {"p": 6, "w": 8, "l": 30}
    meta = simulate_random_dframe(400, vocab_size, ["l"], ["p"], rng)
    feats = rng.normal(size=(len(meta), 16))
    return meta, feats


PIPELINE = dict(pos_sameby=["l"], pos_diffby=["p"], neg_sameby=[], neg_diffby=["l"])
AP_COLUMNS = ["average_precision", "normalized_average_precision"]


def test_average_precision_numba_ap_stage():
    """With the same similarities, both backends agree to float64 rounding."""
    meta, feats = simulated_pipeline_input()

    kwargs = dict(PIPELINE, distance=compute.pairwise_cosine, progress_bar=False)
    fast = average_precision(meta, feats, backend="numba", **kwargs)
    numpy = average_precision(meta, feats, backend="numpy", **kwargs)
    pd.testing.assert_frame_equal(
        fast.drop(columns=AP_COLUMNS), numpy.drop(columns=AP_COLUMNS)
    )
    for col in AP_COLUMNS:
        np.testing.assert_allclose(fast[col], numpy[col], rtol=1e-12, atol=1e-15)


def test_average_precision_numba_similarity():
    """Numba kernels give the NumPy APs up to float32 near-ties."""
    meta, feats = simulated_pipeline_input(1)
    kwargs = dict(PIPELINE, progress_bar=False)
    fast = average_precision(meta, feats, backend="numba", **kwargs)
    numpy = average_precision(meta, feats, backend="numpy", **kwargs)
    np.testing.assert_array_equal(fast["n_total_pairs"], numpy["n_total_pairs"])
    diff = np.abs(fast["average_precision"] - numpy["average_precision"])
    assert (diff.dropna() < 1e-12).mean() > 0.99
    np.testing.assert_allclose(
        fast["average_precision"], numpy["average_precision"], atol=0.05
    )
