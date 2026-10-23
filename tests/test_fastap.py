"""Tests for the Numba similarity kernels and counting-based average precision."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from copairs import fastap, compute
from copairs.map import average_precision
from tests.helpers import run_isolated, simulate_random_dframe
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
    # More profiles than the pairs index: the extra ones have no pairs
    sized = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims, n=130)
    for w, s, z in zip(whole, split, sized):
        np.testing.assert_array_equal(w, s)
        np.testing.assert_array_equal(z, s)


def test_ap_from_pairs_accepts_series_similarities():
    """Pandas similarities are converted to arrays."""
    pos, neg = np.array([[0, 1]]), np.array([[0, 2], [1, 2]])
    pos_sims, neg_sims = np.array([0.5]), np.array([0.1, 0.9])
    expected = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    got = fastap.ap_from_pairs(pos, neg, pd.Series(pos_sims), pd.Series(neg_sims))
    np.testing.assert_array_equal(got[1], expected[1])


MALFORMED_AP_INPUTS = """
import numpy as np
from copairs import fastap
from copairs.fastap.multilabel import multilabel_ap

pos, neg = [[0, 1], [2, 3]], [[0, 2], [1, 3], [0, 3]]
pos_sims, neg_sims = [0.9, 0.8], [0.5, 0.95, 0.1]
cases = [
    # n below an index: the histograms would be written out of bounds
    (IndexError, fastap.ap_from_pairs, (pos, neg, pos_sims, neg_sims), {"n": 2}),
    (IndexError, fastap.ap_from_pairs, ([[0, -1]], neg, [0.9], neg_sims), {}),
    (ValueError, fastap.ap_from_pairs, ([[0, 1]], neg, [], neg_sims), {}),
    (ValueError, fastap.ap_from_pairs, (pos, neg, pos_sims, neg_sims[:2]), {}),
    (ValueError, fastap.ap_from_pairs, (pos, neg, [pos_sims], neg_sims), {}),
    (ValueError, fastap.ap_from_pairs, (pos, neg, pos_sims, neg_sims), {"n": -1}),
    (ValueError, fastap.ap_from_pairs, (np.ravel(pos), neg, [0.9] * 2, neg_sims), {}),
    (IndexError, multilabel_ap, (pos, pos_sims, [1, 1], neg, neg_sims, 2), {}),
    (IndexError, multilabel_ap, (pos, pos_sims, [2], [[0, 9]], [0.1], 4), {}),
    (ValueError, multilabel_ap, (pos, [], [2], neg, neg_sims, 4), {}),
    (ValueError, multilabel_ap, (pos, pos_sims, [2], neg, neg_sims[:1], 4), {}),
    (ValueError, multilabel_ap, (pos, pos_sims, [1], neg, neg_sims, 4), {}),
]
for i, (error, fn, args, kwargs) in enumerate(cases):
    try:
        fn(*args, **kwargs)
    except error:
        pass
    else:
        raise AssertionError(f"case {i}: no {error.__name__}")
"""


def test_ap_kernels_reject_malformed_inputs():
    """Pairs, similarities and n that the kernels would misread raise.

    Runs in a subprocess: the kernels index without bounds checks, so an
    unchecked input can crash the interpreter.
    """
    result = run_isolated(MALFORMED_AP_INPUTS)
    assert result.returncode == 0, (result.returncode, result.stderr)


METRICS = list(fastap.FAST_METRICS)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pair_similarity_matches_generic(metric, dtype):
    """Kernel similarities match copairs' generic functions to float32 rounding."""
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(200, 37)).astype(dtype)
    pairs = rng.integers(200, size=(3000, 2))
    generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 512)
    kernel = fastap.pair_similarity(feats, metric)
    got = kernel(pairs)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, generic, rtol=1e-5, atol=1e-6)
    # find_pairs' uint32 pairs go to the kernels uncopied, with the same result
    np.testing.assert_array_equal(kernel(pairs.astype(np.uint32)), got)


def test_pair_similarity_without_kernel():
    """Metrics without a kernel and callables fall back to the generic path."""
    feats = np.ones((3, 2))
    assert fastap.pair_similarity(feats, "jaccard") is None
    assert fastap.pair_similarity(feats, lambda x, y: x[:, 0]) is None
    generic = compute.get_similarity_fn("jaccard", progress_bar=False)
    assert fastap.similarity_fn(feats, "jaccard", generic) is generic
    with pytest.raises(ValueError, match="no fast kernel"):
        fastap.PairSimilarity(feats, "jaccard")


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


def test_pair_similarity_index_range():
    """Indices outside feats raise IndexError instead of reading out of bounds.

    Runs in a subprocess: an unchecked index can crash the interpreter.
    """
    result = run_isolated(
        "import numpy as np\n"
        "from copairs import fastap, compute\n"
        "feats = np.random.default_rng(0).normal(size=(10, 4))\n"
        "generic = compute.get_similarity_fn('cosine', progress_bar=False)\n"
        "kernels = [fastap.PairSimilarity(feats, m) for m in fastap.FAST_METRICS]\n"
        "kernels.append(lambda p: fastap.similarity_fn(feats, 'cosine', generic)(\n"
        "    feats, p, 8))\n"
        "for kernel in kernels:\n"
        "    for bad in ([[0, 5000]], [[2**31, 0]], [[0, 10]], [[-1, 0]]):\n"
        "        for dtype in (np.int64, np.uint32):\n"
        "            if dtype is np.uint32 and bad[0][0] < 0:\n"
        "                continue\n"
        "            try:\n"
        "                kernel(np.array(bad, dtype=dtype))\n"
        "            except IndexError as exc:\n"
        "                assert 'out of bounds' in str(exc), exc\n"
        "            else:\n"
        "                raise AssertionError(f'no IndexError for {bad} {dtype}')\n"
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize(
    "pairs,match",
    [
        (np.zeros((3, 3), dtype=np.int64), "shape"),
        (np.zeros(4, dtype=np.int64), "shape"),
        (np.zeros((3, 2)), "integer"),
        (np.zeros((3, 2), dtype=bool), "integer"),
    ],
)
def test_pair_similarity_rejects_malformed_pairs(pairs, match):
    """Pairs must be an (n, 2) integer array, not reshaped or cast silently."""
    kernel = fastap.PairSimilarity(np.eye(4), "cosine")
    with pytest.raises(ValueError, match=match):
        kernel(pairs)


def test_pair_similarity_empty_pairs():
    """No pairs give no similarities."""
    kernel = fastap.PairSimilarity(np.eye(4), "euclidean")
    for pairs in ([], np.empty((0, 2), dtype=np.uint32)):
        got = kernel(pairs)
        assert got.shape == (0,) and got.dtype == np.float32


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


def reference_draw_ap(query, reference):
    """AP of each query by fully sorting its rank list.

    Ranked by copairs' float32 key ``1 - similarity``; ties put queries first.
    """
    k = len(query)
    feats = np.concatenate([query, reference]).astype(np.float64)
    feats = (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)
    keys = np.float32(1) - feats[:k] @ feats.T
    keys[np.arange(k), np.arange(k)] = np.inf
    order = np.argsort(keys, axis=1, kind="stable")[:, :-1]
    rel_k = np.nonzero(order < k)[1].reshape(k, k - 1)
    return (np.arange(1, k, dtype=np.float64) / (rel_k + 1)).mean(axis=1)


@pytest.mark.parametrize("k,m,ties", [(2, 5, False), (10, 40, False), (25, 100, True)])
def test_draw_average_precisions(k, m, ties):
    """Batched draw APs equal fully sorted rank lists."""
    rng = np.random.default_rng(4)
    feats = rng.normal(size=(500, 8 if ties else 64)).astype(np.float32)
    if ties:
        feats = np.round(feats)
        feats[:, 0] = 1
    queries = np.stack([rng.choice(500, k, replace=False) for _ in range(30)])
    refs = np.stack([rng.choice(500, m, replace=False) for _ in range(30)])
    expected = np.stack(
        [reference_draw_ap(feats[q], feats[r]) for q, r in zip(queries, refs)]
    )
    got = fastap.draw_average_precisions(
        feats, queries, refs, budget_bytes=4 * (k + m) * 100 * 7
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)


def exact_draws(case, rng):
    """Features whose unit rows and cosines are exact in float32 and float64."""
    if case == "collision":
        # Query 0's negative 2 has cosine 2**-30, its positive 1 has 0: the
        # same float32 key 1 - cosine, so the positive ranks first.
        feats = np.eye(4, dtype=np.float32)
        feats[2, 0] = 2.0**-30  # norm 1 in float64
        return feats, [np.arange(4)], 2
    # Four entries of +-1 per row: cosines are multiples of 1/4 and tie often.
    feats = np.zeros((60, 16), dtype=np.float32)
    for row in feats:
        row[rng.choice(16, 4, replace=False)] = rng.choice([-1, 1], 4)
    return feats, [rng.choice(60, 36, replace=False) for _ in range(5)], 6


@pytest.mark.parametrize("case", ["collision", "ties"])
def test_draw_average_precisions_match_average_precision(case):
    """Draw APs equal average_precision(backend="numba") on the same rank lists."""
    feats, draws, k = exact_draws(case, np.random.default_rng(8))
    got = fastap.draw_average_precisions(
        feats, [r[:k] for r in draws], [r[k:] for r in draws]
    )
    for draw, ap in zip(draws, got):
        # Queries share one label; each reference has its own, so it has no
        # positives and is ranked by the queries as a negative only.
        meta = pd.DataFrame({"g": ["q"] * k + [f"r{i}" for i in draw[k:]]})
        result = average_precision(
            meta, feats[draw], ["g"], [], [], ["g"], progress_bar=False, backend="numba"
        )
        np.testing.assert_array_equal(ap, result["average_precision"].to_numpy()[:k])
    if case == "collision":
        np.testing.assert_array_equal(got, [[1.0, 1.0]])


def test_draw_average_precisions_validation():
    """Too few queries, mismatched draws and degenerate features raise."""
    feats = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2 queries"):
        fastap.draw_average_precisions(feats, [[0]], [[1, 2]])
    with pytest.raises(ValueError, match="n_draws"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2], [3]])
    with pytest.raises(ValueError, match="zero-norm"):
        fastap.draw_average_precisions(np.zeros((4, 2)), [[0, 1]], [[2, 3]])


@pytest.mark.parametrize("bad", [-1, 4])
def test_draw_average_precisions_index_range(bad):
    """Indices outside feats raise instead of wrapping to other rows."""
    feats = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="rows of feats"):
        fastap.draw_average_precisions(feats, [[0, bad]], [[2, 3]])
    with pytest.raises(ValueError, match="rows of feats"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, bad]])


def test_draw_average_precisions_large_norms():
    """Rows whose squared norm overflows float32 keep their cosine ranking."""
    rng = np.random.default_rng(6)
    feats = rng.normal(size=(40, 16)).astype(np.float32)
    queries = np.stack([rng.choice(40, 5, replace=False) for _ in range(10)])
    refs = np.stack([rng.choice(40, 20, replace=False) for _ in range(10)])
    expected = fastap.draw_average_precisions(feats, queries, refs)
    big = feats.copy()
    big[::2] *= np.float32(2.0**66)  # exact scaling; squares exceed float32
    got = fastap.draw_average_precisions(big, queries, refs)
    np.testing.assert_array_equal(got, expected)


def test_draw_average_precisions_nonfinite_normalized():
    """Pre-normalized features are still checked: NaN would break the ranking."""
    feats = np.eye(4, dtype=np.float32)
    feats[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, 3]], normalized=True)
