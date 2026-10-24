"""Tests for the similarity kernels and counting-based average precision.

Tests parametrized over ``AP_BACKENDS`` run the Numba kernels, and the CUDA
ones (:mod:`copairs.fastap.cuda`) where CuPy compiles kernels for a GPU.
"""

import tracemalloc

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from numba.core.dispatcher import Dispatcher

from copairs import fastap, compute, matching, _numba_cache
from copairs.map import multilabel, average_precision
from tests.helpers import (
    fake_gpu,
    run_on_copy,
    copy_copairs,
    run_isolated,
    simulate_random_dframe,
)
from tests.markers import AP_BACKENDS, needs_gpu, needs_cupy
from copairs.fastap import draws, ranking, similarity
from copairs.fastap import multilabel as fast_multilabel
from copairs._backend import backend_module
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


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("ties", [False, True])
def test_ap_from_pairs_matches_rank_lists(backend, ties):
    """Counting AP equals rank-list AP, including tied similarities."""
    rng = np.random.default_rng(0)
    pos, neg = random_pairs(rng, 300, 900, 20_000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    if ties:
        pos_sims = np.round(pos_sims * 8) / 8
        neg_sims = np.round(neg_sims * 8) / 8
    expected = rank_list_ap(pos, neg, pos_sims, neg_sims)
    got = backend_module(backend).ap_from_pairs(pos, neg, pos_sims, neg_sims)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[2], expected[2])
    assert got[2].dtype == expected[2].dtype
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_ap_from_pairs_edge_cases(backend):
    """Profiles without positives get NaN; NaN similarities rank last."""
    rng = np.random.default_rng(1)
    pos, neg = random_pairs(rng, 50, 40, 500)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    pos_sims[::7] = np.nan
    neg_sims[::5] = np.nan
    expected = rank_list_ap(pos, neg, pos_sims, neg_sims)
    got = backend_module(backend).ap_from_pairs(pos, neg, pos_sims, neg_sims)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(np.isnan(got[1]), np.isnan(expected[1]))
    assert np.isnan(got[1]).any()
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_ap_from_pairs_histogram_chunks(backend):
    """Splitting negatives over per-thread histograms leaves APs unchanged."""
    rng = np.random.default_rng(7)
    pos, neg = random_pairs(rng, 100, 300, 5000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    kernels = backend_module(backend)
    whole = kernels.ap_from_pairs(pos, neg, pos_sims, neg_sims, budget_bytes=1)
    split = kernels.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    # More profiles than the pairs index: the extra ones have no pairs
    sized = kernels.ap_from_pairs(pos, neg, pos_sims, neg_sims, n=130)
    for w, s, z in zip(whole, split, sized):
        np.testing.assert_array_equal(w, s)
        np.testing.assert_array_equal(z, s)


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("n", [None, 0, 3])
@pytest.mark.parametrize("empty", [[], np.empty((0, 2), dtype=np.uint32)])
def test_ap_from_pairs_without_pairs(n, empty, backend):
    """No pairs give empty results, also with no profiles to size histograms by."""
    kernels = backend_module(backend)
    paired_ix, ap, null_confs = kernels.ap_from_pairs(empty, empty, [], [], n=n)
    assert paired_ix.shape == (0,) and paired_ix.dtype == np.int64
    assert ap.shape == (0,) and ap.dtype == np.float64
    assert null_confs.shape == (0, 2) and null_confs.dtype == np.uint32


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_ap_from_pairs_accepts_series_similarities(backend):
    """Pandas similarities are converted to arrays."""
    pos, neg = np.array([[0, 1]]), np.array([[0, 2], [1, 2]])
    pos_sims, neg_sims = np.array([0.5]), np.array([0.1, 0.9])
    kernels = backend_module(backend)
    expected = kernels.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    got = kernels.ap_from_pairs(pos, neg, pd.Series(pos_sims), pd.Series(neg_sims))
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


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pair_similarity_matches_generic(backend, metric, dtype):
    """Kernel similarities match copairs' generic functions to float32 rounding."""
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(200, 37)).astype(dtype)
    pairs = rng.integers(200, size=(3000, 2))
    generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 512)
    kernel = backend_module(backend).pair_similarity(feats, metric)
    got = kernel(pairs)
    assert isinstance(got, np.ndarray) and got.dtype == np.float32
    np.testing.assert_allclose(got, generic, rtol=1e-5, atol=1e-6)
    # find_pairs' uint32 pairs go to the kernels uncopied, with the same result
    np.testing.assert_array_equal(kernel(pairs.astype(np.uint32)), got)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_serial_pair_similarity_gives_the_parallel_bits(metric, dtype):
    """The serial kernel (the nulls' threads') gives the parallel kernel's bits.

    The permutation null re-ranks profiles by them and must reproduce the
    AP stage's APs.
    """
    rng = np.random.default_rng(4)
    for d in (1, 7, 37, 300):
        feats = (rng.normal(size=(200, d)) * 3).astype(dtype)
        pairs = rng.integers(200, size=(5000, 2))
        serial = fastap.pair_similarity(feats, metric, parallel=False)(pairs)
        parallel = fastap.pair_similarity(feats, metric)(pairs)
        np.testing.assert_array_equal(serial.view(np.uint32), parallel.view(np.uint32))


def test_pair_similarity_without_kernel():
    """Metrics without a kernel and callables fall back to the generic path."""
    feats = np.ones((3, 2))
    assert fastap.pair_similarity(feats, "jaccard") is None
    assert fastap.pair_similarity(feats, lambda x, y: x[:, 0]) is None
    generic = compute.get_similarity_fn("jaccard", progress_bar=False)
    assert fastap.similarity_fn(feats, "jaccard", generic) is generic
    with pytest.raises(ValueError, match="no fast kernel"):
        fastap.PairSimilarity(feats, "jaccard")


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_similarity_fn_is_bound_to_its_feats(backend):
    """The kernel replacement scores the feats it was built for, no others."""
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(20, 5))
    pairs = np.array([[0, 1], [2, 3]])
    generic = compute.get_similarity_fn("cosine", progress_bar=False)
    fn = backend_module(backend).similarity_fn(feats, "cosine", generic)
    assert fn is not generic
    np.testing.assert_allclose(fn(feats, pairs, 1), generic(feats, pairs, 1), rtol=1e-6)
    for other in (feats.copy(), feats[::-1], rng.normal(size=(20, 5))):
        with pytest.raises(ValueError, match="built for"):
            fn(other, pairs, 1)


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_pair_similarity_nonfinite_features(backend, metric):
    """Inf - inf differences give NaN similarities, as in the generic functions."""
    feats = np.array([[np.inf, 1.0, 0.0], [np.inf, 0.0, 0.0], [1.0, 2.0, 3.0]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    with np.errstate(invalid="ignore"):
        generic = compute.get_similarity_fn(metric, progress_bar=False)(feats, pairs, 8)
    got = backend_module(backend).pair_similarity(feats, metric)(pairs)
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
@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_pair_similarity_rejects_malformed_pairs(backend, pairs, match):
    """Pairs must be an (n, 2) integer array, not reshaped or cast silently."""
    kernel = backend_module(backend).pair_similarity(np.eye(4), "cosine")
    with pytest.raises(ValueError, match=match):
        kernel(pairs)


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_pair_similarity_empty_pairs(backend):
    """No pairs give no similarities."""
    kernel = backend_module(backend).pair_similarity(np.eye(4), "euclidean")
    for pairs in ([], np.empty((0, 2), dtype=np.uint32)):
        got = kernel(pairs)
        assert got.shape == (0,) and got.dtype == np.float32


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_pair_similarity_float32_overflow(backend):
    """Rows whose float32 norm overflows get their cosine, not the generic 0."""
    feats = np.array(
        [[1e20, 2e20, 0.0], [2e20, 1e20, 0.0], [1.0, 2.0, 0.0]], np.float32
    )
    pairs = np.array([[0, 1], [0, 2]])
    got = backend_module(backend).pair_similarity(feats, "cosine")(pairs)
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


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_average_precision_fast_ap_stage(backend):
    """With the same similarities, the fast and NumPy APs agree to float64 rounding."""
    meta, feats = simulated_pipeline_input()

    kwargs = dict(PIPELINE, distance=compute.pairwise_cosine, progress_bar=False)
    fast = average_precision(meta, feats, backend=backend, **kwargs)
    numpy = average_precision(meta, feats, backend="numpy", **kwargs)
    pd.testing.assert_frame_equal(
        fast.drop(columns=AP_COLUMNS), numpy.drop(columns=AP_COLUMNS)
    )
    for col in AP_COLUMNS:
        np.testing.assert_allclose(fast[col], numpy[col], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_average_precision_fast_similarity(backend):
    """The kernels give the NumPy APs up to float32 near-ties."""
    meta, feats = simulated_pipeline_input(1)
    kwargs = dict(PIPELINE, progress_bar=False)
    fast = average_precision(meta, feats, backend=backend, **kwargs)
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


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("k,m,ties", [(2, 5, False), (10, 40, False), (25, 100, True)])
def test_draw_average_precisions(backend, k, m, ties):
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
        feats, queries, refs, budget_bytes=4 * (k + m) * 100 * 7, backend=backend
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


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("case", ["collision", "ties"])
def test_draw_average_precisions_match_average_precision(backend, case):
    """Draw APs equal average_precision(backend="numba") on the same rank lists."""
    feats, draws, k = exact_draws(case, np.random.default_rng(8))
    got = fastap.draw_average_precisions(
        feats, [r[:k] for r in draws], [r[k:] for r in draws], backend=backend
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
    with pytest.raises(ValueError, match="'numba' or 'cuda'"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, 3]], backend="numpy")


def test_draw_average_precisions_validate_before_the_backend(monkeypatch):
    """Bad draws raise ValueError before the backend is looked up.

    Under "cuda" the lookup checks the GPU, whose RuntimeError (no usable
    GPU) would otherwise hide the input error.
    """

    def looked_up(backend):
        raise AssertionError(f"backend {backend!r} looked up")

    monkeypatch.setattr(draws, "backend_module", looked_up)
    feats = np.eye(4, dtype=np.float32)
    for args, match in (
        ((feats, [[0]], [[1, 2]]), "at least 2 queries"),
        ((feats, [[0, 1]], [[2], [3]]), "n_draws"),
        ((feats, [[0, 4]], [[2, 3]]), "rows of feats"),
        ((np.zeros((4, 2)), [[0, 1]], [[2, 3]]), "zero-norm"),
        ((np.full((4, 2), np.nan), [[0, 1]], [[2, 3]]), "non-finite"),
    ):
        with pytest.raises(ValueError, match=match):
            fastap.draw_average_precisions(*args, backend="cuda")


@pytest.mark.parametrize("bad", [-1, 4])
def test_draw_average_precisions_index_range(bad):
    """Indices outside feats raise instead of wrapping to other rows."""
    feats = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="rows of feats"):
        fastap.draw_average_precisions(feats, [[0, bad]], [[2, 3]])
    with pytest.raises(ValueError, match="rows of feats"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, bad]])


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_draw_average_precisions_large_norms(backend):
    """Rows whose squared norm overflows float32 keep their cosine ranking."""
    rng = np.random.default_rng(6)
    feats = rng.normal(size=(40, 16)).astype(np.float32)
    queries = np.stack([rng.choice(40, 5, replace=False) for _ in range(10)])
    refs = np.stack([rng.choice(40, 20, replace=False) for _ in range(10)])
    expected = fastap.draw_average_precisions(feats, queries, refs, backend=backend)
    big = feats.copy()
    big[::2] *= np.float32(2.0**66)  # exact scaling; squares exceed float32
    got = fastap.draw_average_precisions(big, queries, refs, backend=backend)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("scale", [1e40, 1e-50])
def test_draw_average_precisions_float64_beyond_float32(backend, scale):
    """float64 rows that float32 cannot hold keep their direction."""
    rng = np.random.default_rng(12)
    feats = rng.normal(size=(40, 16))
    queries = np.stack([rng.choice(40, 5, replace=False) for _ in range(10)])
    refs = np.stack([rng.choice(40, 20, replace=False) for _ in range(10)])
    expected = fastap.draw_average_precisions(feats, queries, refs, backend=backend)
    got = fastap.draw_average_precisions(feats * scale, queries, refs, backend=backend)
    np.testing.assert_array_equal(got, expected)


def test_draw_average_precisions_chunks_do_not_overlap():
    """Each chunk's gathered rows are freed before the next chunk's are built.

    The peak with 40 chunks exceeds the peak with one chunk only by the
    index and output arrays that grow with the number of draws.
    """
    rng = np.random.default_rng(11)
    feats = rng.normal(size=(50, 256)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    k, m, step = 4, 46, 10
    budget = step * 4 * (k + m) * (feats.shape[1] + k)

    def peak(n_draws):
        q = np.stack([rng.choice(50, k, replace=False) for _ in range(n_draws)])
        r = np.stack([rng.choice(50, m, replace=False) for _ in range(n_draws)])
        args = (feats, q, r)
        fastap.draw_average_precisions(*args, normalized=True, budget_bytes=budget)
        tracemalloc.start()
        try:
            fastap.draw_average_precisions(*args, normalized=True, budget_bytes=budget)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    one, many = peak(step), peak(40 * step)
    rows = step * (k + m) * feats.shape[1] * 4  # one chunk's gathered features
    grown = 39 * step * (k + m + k) * 8  # idx and out of the extra draws
    assert many - one < grown + rows / 2, (one, many, grown, rows)


def test_draw_average_precisions_nonfinite_normalized():
    """Pre-normalized features are still checked: NaN would break the ranking."""
    feats = np.eye(4, dtype=np.float32)
    feats[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, 3]], normalized=True)


@pytest.mark.parametrize("normalized", [False, True])
def test_draw_average_precisions_touch_only_drawn_rows(normalized):
    """Only rows that draws use are converted, so a few draws stay cheap.

    APs equal those from normalizing every row first.
    """
    rng = np.random.default_rng(9)
    feats = rng.normal(size=(100_000, 64))  # float64: 51 MB
    if normalized:
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    queries = rng.choice(len(feats), size=(20, 4))
    refs = rng.choice(len(feats), size=(20, 30))
    expected = fastap.draw_average_precisions(
        fastap.draws.unit_rows(feats, normalized), queries, refs, normalized=True
    )
    tracemalloc.start()
    try:
        got = fastap.draw_average_precisions(feats, queries, refs, normalized)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(got, expected)
    assert peak < feats.nbytes / 10, peak


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_draw_average_precisions_check_drawn_rows(backend):
    """Drawn rows that are zero or non-finite still raise; others are unused."""
    feats = np.eye(6, dtype=np.float32)
    feats[1] = 0
    feats[4, 0] = np.nan
    with pytest.raises(ValueError, match="zero-norm"):
        fastap.draw_average_precisions(feats, [[0, 1]], [[2, 3]])
    with pytest.raises(ValueError, match="non-finite"):
        fastap.draw_average_precisions(feats, [[0, 2]], [[4, 3]], normalized=True)
    got = fastap.draw_average_precisions(feats, [[0, 2]], [[3, 5]], backend=backend)
    np.testing.assert_array_equal(got, [[1.0, 1.0]])


@needs_gpu
def test_ap_from_pairs_cuda_negative_chunks():
    """Negatives streamed to the GPU in chunks give the APs of one pass.

    Similarities on the host, on the GPU or computed in the chunks
    (DeviceSims) give the same result.
    """
    cuda = backend_module("cuda")
    rng = np.random.default_rng(6)
    pos, neg = random_pairs(rng, 200, 600, 10_000)
    feats = rng.normal(size=(200, 7))
    kernel = cuda.pair_similarity(feats, "cosine", on_device=True)
    lazy_pos, lazy_neg = kernel(pos), kernel(neg)
    pos_sims, neg_sims = lazy_pos.get(), lazy_neg.get()
    whole = cuda.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    expected = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    for g, e in zip(whole, expected):
        np.testing.assert_array_equal(g, e)
    for sims in (neg_sims, cuda.cp.asarray(neg_sims), lazy_neg):
        got = cuda.ap_from_pairs(pos, neg, lazy_pos, sims, budget_bytes=16 * 999)
        for g, w in zip(got, whole):
            assert isinstance(g, np.ndarray)
            np.testing.assert_array_equal(g, w)
    with pytest.raises(ValueError, match="other pairs"):
        cuda.ap_from_pairs(pos, neg[::-1], lazy_pos, lazy_neg)


@needs_gpu
@pytest.mark.parametrize("ties", [False, True])
def test_cuda_ap_equals_numba_bitwise(ties):
    """Given the same similarities, CUDA APs equal Numba's bit for bit.

    Counts are exact and both sum each AP in the same order; NaN similarities
    and ties included.
    """
    rng = np.random.default_rng(13)
    pos, neg = random_pairs(rng, 500, 3000, 50_000)
    pos_sims = rng.random(len(pos)).astype(np.float32)
    neg_sims = rng.random(len(neg)).astype(np.float32)
    if ties:
        pos_sims, neg_sims = np.round(pos_sims * 8) / 8, np.round(neg_sims * 8) / 8
    pos_sims[::11], neg_sims[::13] = np.nan, np.nan
    expected = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)
    got = backend_module("cuda").ap_from_pairs(pos, neg, pos_sims, neg_sims)
    for g, e in zip(got, expected):
        assert g.dtype == e.dtype and g.tobytes() == e.tobytes()
    meta, feats = simulated_pipeline_input(2)
    kwargs = dict(PIPELINE, distance=compute.pairwise_cosine, progress_bar=False)
    pd.testing.assert_frame_equal(
        average_precision(meta, feats, backend="cuda", **kwargs),
        average_precision(meta, feats, backend="numba", **kwargs),
        check_exact=True,
    )


@needs_gpu
def test_cuda_similarities_computed_in_the_ap_count(monkeypatch):
    """average_precision leaves the kernels' similarities to the GPU AP count."""
    cuda = backend_module("cuda")
    feats = np.random.default_rng(14).normal(size=(30, 6))
    generic = compute.get_similarity_fn("cosine", progress_bar=False)
    pairs = np.array([[0, 1], [2, 3]])
    sims = cuda.similarity_fn(feats, "cosine", generic)(feats, pairs, 1)
    assert isinstance(sims, cuda.DeviceSims) and len(sims) == 2
    host = sims.get()
    np.testing.assert_array_equal(np.asarray(sims), host)
    np.testing.assert_allclose(host, generic(feats, pairs, 1), rtol=1e-6)

    seen = []
    ap_from_pairs = cuda.ap_from_pairs

    def record(pos_pairs, neg_pairs, pos_sims, neg_sims, **kwargs):
        seen.append((type(pos_sims), type(neg_sims)))
        return ap_from_pairs(pos_pairs, neg_pairs, pos_sims, neg_sims, **kwargs)

    monkeypatch.setattr(cuda, "ap_from_pairs", record)
    meta, feats = simulated_pipeline_input(3)
    average_precision(meta, feats, **PIPELINE, progress_bar=False, backend="cuda")
    assert seen == [(cuda.DeviceSims, cuda.DeviceSims)]


@needs_gpu
@pytest.mark.parametrize("extra", [0, 1])
def test_draw_average_precisions_shared_memory_limit(extra):
    """Draws up to the CUDA kernel's shared memory run on the GPU, larger on Numba.

    Either way their APs equal the Numba backend's up to float32 near-ties.
    """
    cuda = backend_module("cuda")
    k, m = cuda._MAX_DRAW_QUERIES + extra, 16
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(k + m, 8)).astype(np.float32)
    queries, refs = np.arange(k)[None], np.arange(k, k + m)[None]
    got = fastap.draw_average_precisions(feats, queries, refs, backend="cuda")
    expected = fastap.draw_average_precisions(feats, queries, refs)
    if extra:
        np.testing.assert_array_equal(got, expected)
    else:
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


@needs_cupy
def test_cuda_ap_from_pairs_checks_device_sims(monkeypatch):
    """CuPy similarities, like NumPy ones, must be one per pair."""
    cp, _ = fake_gpu(monkeypatch)
    cuda = backend_module("cuda")
    pos, neg = np.array([[0, 1]]), np.array([[0, 2], [1, 2]])
    for sims in ([0.5], np.array([0.5]), cp.asarray(np.array([0.5]))):
        with pytest.raises(ValueError, match="one similarity per pair"):
            cuda.ap_from_pairs(pos, neg, [0.9], sims)


@needs_cupy
def test_cuda_pairs_cast_to_uint32_on_the_host(monkeypatch):
    """Pairs reach the GPU as uint32, 8 bytes per pair, whatever their host dtype."""
    cp, launches = fake_gpu(monkeypatch)
    cuda = backend_module("cuda")
    rng = np.random.default_rng(15)
    pos, neg = random_pairs(rng, 50, 100, 2000)
    feats = rng.normal(size=(50, 4))
    for metric in ("cosine", "euclidean"):
        cp.allocated.clear()
        cuda.pair_similarity(feats, metric)(neg.astype(np.int64))
        assert max(cp.allocated) <= 8 * len(neg), cp.allocated
    cp.allocated.clear()
    launches.clear()
    cuda.ap_from_pairs(pos, neg, rng.random(len(pos)), rng.random(len(neg)))
    assert max(cp.allocated) <= 8 * len(neg), cp.allocated
    for _, args in launches:
        pairs = [a for a in args if isinstance(a, np.ndarray) and a.ndim == 2]
        assert all(a.dtype == np.uint32 for a in pairs)


@needs_cupy
def test_cuda_ap_stage_streams_negatives_within_budget(monkeypatch):
    """The GPU AP count holds at most budget_bytes of negatives at a time.

    Their similarities are computed chunk by chunk, not all beforehand.
    """
    cp, launches = fake_gpu(monkeypatch)
    cuda = backend_module("cuda")
    rng = np.random.default_rng(16)
    pos, neg = random_pairs(rng, 40, 10, 5000)
    feats = rng.normal(size=(40, 4))
    generic = compute.get_similarity_fn("cosine", progress_bar=False)
    similarity = cuda.similarity_fn(feats, "cosine", generic)
    cp.allocated.clear()
    pos_sims, neg_sims = similarity(feats, pos, 1), similarity(feats, neg, 1)
    assert not cp.allocated, "no similarities computed before the AP count"
    budget = 16 * 100
    cuda.ap_from_pairs(pos, neg, pos_sims, neg_sims, budget_bytes=budget)
    assert max(cp.allocated) <= budget, cp.allocated
    assert sum(name == "negative_hist" for name, _ in launches) == len(neg) // 100


@needs_cupy
@pytest.mark.parametrize("resident", [True, False])
def test_cuda_draws_within_budget(monkeypatch, resident):
    """GPU draws hold one chunk's rows, similarities and APs within the budget.

    APs go to the host chunk by chunk; the drawn rows stay on the GPU only
    when they take at most half of the budget.
    """
    cp, launches = fake_gpu(monkeypatch)
    rng = np.random.default_rng(17)
    k, m, n_draws = 4, 12, 200  # all APs: 6400 bytes
    feats = rng.normal(size=(40, 8)).astype(np.float32)
    queries = np.stack([rng.choice(40, k, replace=False) for _ in range(n_draws)])
    refs = np.stack([rng.choice(40, m, replace=False) for _ in range(n_draws)])
    per_draw = 4 * (k + m) * (8 + k)
    budget = 3 * per_draw + (2 * feats.nbytes if resident else 0)
    got = fastap.draw_average_precisions(
        feats, queries, refs, budget_bytes=budget, backend="cuda"
    )
    assert isinstance(got, np.ndarray) and got.shape == (n_draws, k)
    assert max(cp.allocated) <= budget, cp.allocated
    assert (feats.nbytes in cp.allocated) == resident
    assert sum(name == "draw_ap" for name, _ in launches) > 1


# APs of the kernels that compile ranking.py's functions, and their cache hits,
# for a copy of copairs in the directory argv[1].
_CACHE_PROBE = """
import sys, json
import numpy as np
import copairs
from copairs import fastap
from copairs.fastap import draws, multilabel

assert copairs.__file__.startswith(sys.argv[1]), copairs.__file__
rng = np.random.default_rng(0)
feats = rng.normal(size=(40, 8))
queries = np.array([rng.choice(40, 4, replace=False) for _ in range(20)])
refs = np.array([rng.choice(40, 6, replace=False) for _ in range(20)])
pos = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
neg = np.array([[0, 4], [1, 5], [2, 6], [3, 7], [0, 5]])
pos_sims, neg_sims = rng.random(4), rng.random(5)
pairs_ap = fastap.ap_from_pairs(pos, neg, pos_sims, neg_sims)[1]
out = {
    "pairs": np.nan_to_num(pairs_ap, nan=-1).tolist(),
    "draws": fastap.draw_average_precisions(feats, queries, refs).tolist(),
    "multilabel": multilabel.multilabel_ap(pos, pos_sims, [2, 2], neg, neg_sims, 8)[
        2
    ].tolist(),
    "hits": [
        sum(k.stats.cache_hits.values()) for k in (draws._row_ap, multilabel._rows_ap)
    ],
}
print(json.dumps(out))
"""


@pytest.mark.parametrize(
    "path,old,new",
    [
        (
            "fastap/ranking.py",
            "    return acc / num_pos\n",
            "    return 0.5 * acc / num_pos\n",
        ),
        # Keys lie in [0, 2]: every positive now ranks ahead of every negative.
        ("_search.py", "v <= key)", "v <= key + 2.0)"),
    ],
    ids=["ranking", "upper_bound"],
)
def test_numba_cache_follows_edits_to_compiled_files(tmp_path, path, old, new):
    """Cached AP kernels follow edits to the files of the functions they compile.

    Numba checks a cached kernel only against its own file. The draw and
    multilabel AP kernels compile ranking.py's ``ap_from_counts``, and every
    AP kernel the binary search of _search.py, shared with the Numba nulls.
    After an edit, the next process gives the APs of a process with an empty
    cache, while a process after an unchanged one loads the kernels from the
    cache.
    """
    copy_copairs(tmp_path)
    before = run_on_copy(tmp_path, _CACHE_PROBE)
    assert run_on_copy(tmp_path, _CACHE_PROBE)["hits"] == [1, 1]
    source = tmp_path / "copairs" / path
    text = source.read_text()
    assert text.count(old) == 1
    source.write_text(text.replace(old, new))
    after = run_on_copy(tmp_path, _CACHE_PROBE)
    fresh = run_on_copy(tmp_path, _CACHE_PROBE, cache="empty_cache")
    for key in ("pairs", "draws", "multilabel"):
        assert after[key] == fresh[key] != before[key], key


def test_ap_stage_kernels_launch_under_the_launch_lock(monkeypatch):
    """Every parallel kernel of the AP stage runs under the nulls' launch lock.

    Numba's workqueue layer aborts the process on two parallel launches at
    once, e.g. a user thread's average_precision next to a permutation
    null's draws, and the OpenMP layer starts a team per launching thread.
    """
    held, patched = {}, []
    for module in (similarity, ranking, fast_multilabel, draws):
        for name, kernel in list(vars(module).items()):
            if isinstance(kernel, Dispatcher) and kernel.targetoptions.get("parallel"):

                def spy(*args, kernel=kernel, name=name):
                    held.setdefault(name, set()).add(_numba_cache._LAUNCH.locked())
                    return kernel(*args)

                monkeypatch.setattr(module, name, spy)
                patched.append(name)
    rng = np.random.default_rng(9)
    meta = pd.DataFrame({"g": np.repeat(np.arange(6), 4), "plate": np.arange(24) % 2})
    meta["labels"] = [[f"L{g}", f"L{g + 1}"] for g in meta["g"]]
    feats = rng.normal(size=(24, 5))
    for distance in ("cosine", "euclidean"):
        average_precision(
            meta, feats, ["g"], [], [], ["g"], progress_bar=False,
            backend="numba", distance=distance,
        )  # fmt: skip
    multilabel.average_precision(
        meta, feats, ["labels"], [], [], ["labels"], multilabel_col="labels",
        progress_bar=False, backend="numba",
    )  # fmt: skip
    matching.find_pairs_multilabel(meta, [], ["labels"], "labels", backend="numba")
    fastap.draw_average_precisions(
        feats, [[0, 1], [2, 3]], [[4, 5, 6], [7, 8, 9]], backend="numba"
    )
    assert len(patched) == 8 and held == {name: {True} for name in patched}
