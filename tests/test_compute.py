"""Test pairwise distance calculation functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist as scipy_cdist

from copairs import compute

SEED = 0
rng = np.random.default_rng(SEED)


def corrcoef_naive(feats, pairs):
    """Compute correlation coefficient between pairs of features."""
    corr = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        corr[pos] = np.corrcoef(feats[i], feats[j])[0, 1]
    return corr


def cosine_naive(feats, pairs):
    """Compute cosine similarity between pairs of features."""
    cosine = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        a, b = feats[i], feats[j]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine[pos] = a.dot(b) / (norm_a * norm_b)
    return cosine


def euclidean_naive(feats, pairs):
    """Compute euclidean similarity between pairs of features."""
    euclidean_sim = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        dist = np.linalg.norm(feats[i] - feats[j])
        euclidean_sim[pos] = 1 / (1 + dist)
    return euclidean_sim


def abs_cosine_naive(feats, pairs):
    """Compute absolute cosine similarity between pairs of features."""
    return np.abs(cosine_naive(feats, pairs))


def manhattan_naive(feats, pairs):
    """Compute inverse Manhattan similarity between pairs of features."""
    manhattan_sim = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        dist = np.sum(np.abs(feats[i] - feats[j]))
        manhattan_sim[pos] = 1 / (1 + dist)
    return manhattan_sim


def chebyshev_naive(feats, pairs):
    """Compute inverse Chebyshev similarity between pairs of features."""
    chebyshev_sim = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        dist = np.max(np.abs(feats[i] - feats[j]))
        chebyshev_sim[pos] = 1 / (1 + dist)
    return chebyshev_sim


def jaccard_naive(feats, pairs):
    """Compute Jaccard similarity between pairs of binary features."""
    jaccard_sim = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        intersection = np.sum(np.minimum(feats[i], feats[j]))
        union = np.sum(np.maximum(feats[i], feats[j]))
        jaccard_sim[pos] = 1 - (1 - intersection / union) if union > 0 else 1.0
    return jaccard_sim


def hamming_naive(feats, pairs):
    """Compute Hamming similarity between pairs of binary features."""
    hamming_sim = np.empty((len(pairs),))
    for pos, (i, j) in enumerate(pairs):
        dist = np.sum(feats[i] != feats[j]) / len(feats[i])
        hamming_sim[pos] = 1 - dist
    return hamming_sim


class UnhashableMetric:
    """Callable metric that cannot be used in set membership checks."""

    __hash__ = None

    def __call__(self, x, y):
        """Return the city-block distance between two vectors."""
        return np.sum(np.abs(x - y))


def test_corrcoef():
    """Test correlation coefficient computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    corr_gt = corrcoef_naive(feats, pairs)
    corr_fn = compute.get_similarity_fn("correlation")
    corr = corr_fn(feats, pairs, batch_size)
    assert np.allclose(corr_gt, corr)


def test_cosine():
    """Test cosine similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    cosine_gt = cosine_naive(feats, pairs)
    cosine_fn = compute.get_similarity_fn("cosine")
    cosine = cosine_fn(feats, pairs, batch_size)
    assert np.allclose(cosine_gt, cosine)


def test_euclidean():
    """Test euclidean similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    euclidean_gt = euclidean_naive(feats, pairs)
    euclidean_fn = compute.get_similarity_fn("euclidean")
    euclidean = euclidean_fn(feats, pairs, batch_size)
    assert np.allclose(euclidean_gt, euclidean)


def test_abs_cosine():
    """Test absolute cosine similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    abs_cosine_gt = abs_cosine_naive(feats, pairs)
    abs_cosine_fn = compute.get_similarity_fn("abs_cosine")
    abs_cosine = abs_cosine_fn(feats, pairs, batch_size)
    assert np.allclose(abs_cosine_gt, abs_cosine)


def test_manhattan():
    """Test Manhattan similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    manhattan_gt = manhattan_naive(feats, pairs)
    manhattan_fn = compute.get_similarity_fn("manhattan")
    manhattan = manhattan_fn(feats, pairs, batch_size)
    assert np.allclose(manhattan_gt, manhattan)


def test_chebyshev():
    """Test Chebyshev similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    chebyshev_gt = chebyshev_naive(feats, pairs)
    chebyshev_fn = compute.get_similarity_fn("chebyshev")
    chebyshev = chebyshev_fn(feats, pairs, batch_size)
    assert np.allclose(chebyshev_gt, chebyshev)


def test_jaccard():
    """Test Jaccard similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.integers(0, 2, [n_samples, n_feats])  # Binary data for Jaccard
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    jaccard_gt = jaccard_naive(feats, pairs)
    jaccard_fn = compute.get_similarity_fn("jaccard")
    jaccard = jaccard_fn(feats, pairs, batch_size)
    assert np.allclose(jaccard_gt, jaccard)


def test_hamming():
    """Test Hamming similarity computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.integers(0, 2, [n_samples, n_feats])  # Binary data for Hamming
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    hamming_gt = hamming_naive(feats, pairs)
    hamming_fn = compute.get_similarity_fn("hamming")
    hamming = hamming_fn(feats, pairs, batch_size)
    assert np.allclose(hamming_gt, hamming)


@pytest.mark.parametrize(
    "metric",
    ["hamming", "jaccard", "euclidean", "cityblock", "manhattan", "sqeuclidean"],
)
@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.int16])
def test_cdist_diag_sim_rowwise_matches_scipy(metric, dtype):
    """Direct row-wise kernels match the diagonal of SciPy's cdist."""
    x_sample = np.array([[0, 0, 0, 0], [1, 0, 2, -1], [3, 4, 0, 2]], dtype=dtype)
    y_sample = np.array([[0, 0, 0, 0], [0, 1, 2, -2], [1, 4, 2, 2]], dtype=dtype)
    scipy_metric = "cityblock" if metric == "manhattan" else metric
    distance = np.diag(scipy_cdist(x_sample, y_sample, metric=scipy_metric))
    expected = 1 - distance if metric in {"hamming", "jaccard"} else 1 / (1 + distance)

    actual = compute._cdist_diag_sim(x_sample, y_sample, metric)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "metric",
    ["hamming", "jaccard", "euclidean", "cityblock", "manhattan", "sqeuclidean"],
)
def test_cdist_diag_sim_unsigned_matches_scipy(metric):
    """Direct row-wise kernels avoid unsigned subtraction underflow."""
    x_sample = np.array([[0, 65535, 1, 32768], [65535, 0, 100, 1]], dtype=np.uint16)
    y_sample = np.array([[65535, 0, 2, 32767], [0, 65535, 99, 1]], dtype=np.uint16)
    scipy_metric = "cityblock" if metric == "manhattan" else metric
    distance = np.diag(scipy_cdist(x_sample, y_sample, metric=scipy_metric))
    expected = 1 - distance if metric in {"hamming", "jaccard"} else 1 / (1 + distance)

    actual = compute._cdist_diag_sim(x_sample, y_sample, metric)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize("shape", [(0, 4), (3, 0)])
@pytest.mark.parametrize(
    "metric",
    ["hamming", "jaccard", "euclidean", "cityblock", "manhattan", "sqeuclidean"],
)
def test_cdist_diag_sim_empty_shapes_match_scipy(metric, shape):
    """Supported empty row and feature dimensions preserve SciPy results."""
    x_sample = np.empty(shape)
    y_sample = np.empty(shape)
    scipy_metric = "cityblock" if metric == "manhattan" else metric
    distance = np.diag(scipy_cdist(x_sample, y_sample, metric=scipy_metric))
    expected = 1 - distance if metric in {"hamming", "jaccard"} else 1 / (1 + distance)

    actual = compute._cdist_diag_sim(x_sample, y_sample, metric)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.bool_, np.float64])
def test_cdist_diag_sim_jaccard_matches_scipy(dtype):
    """Row-wise Jaccard preserves zero-vector and numeric-input semantics."""
    x_sample = np.array(
        [[0, 0, 0, 0], [1, 0, 2, -1], [np.nan, 0, np.inf, 1]], dtype=dtype
    )
    y_sample = np.array(
        [[0, 0, 0, 0], [1, 3, 4, -1], [np.nan, 0, -np.inf, 0]], dtype=dtype
    )
    expected = 1 - np.diag(scipy_cdist(x_sample, y_sample, metric="jaccard"))

    actual = compute._cdist_diag_sim(x_sample, y_sample, "jaccard")

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("metric", ["hamming", "euclidean", "cityblock", "sqeuclidean"])
def test_cdist_diag_sim_nonfinite_matches_scipy(metric):
    """Row-wise kernels preserve SciPy results for nonfinite values."""
    x_sample = np.array([[np.nan, 1], [np.inf, 0], [-np.inf, 2]])
    y_sample = np.array([[np.nan, 2], [np.inf, 1], [np.inf, 2]])
    distance = np.diag(scipy_cdist(x_sample, y_sample, metric=metric))
    expected = 1 - distance if metric == "hamming" else 1 / (1 + distance)

    actual = compute._cdist_diag_sim(x_sample, y_sample, metric)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_cdist_diag_sim_supported_metrics_do_not_call_cdist(monkeypatch):
    """Supported metrics never allocate a full cdist result."""
    x_sample = np.array([[False, True], [True, True]])
    y_sample = np.array([[False, False], [True, False]])

    def fail_cdist(*args, **kwargs):
        raise AssertionError("row-wise metric unexpectedly used cdist")

    monkeypatch.setattr(compute, "cdist", fail_cdist)
    for metric in [
        "hamming",
        "jaccard",
        "euclidean",
        "cityblock",
        "manhattan",
        "sqeuclidean",
    ]:
        compute._cdist_diag_sim(x_sample, y_sample, metric)


@pytest.mark.parametrize(
    "metric",
    ["canberra", pytest.param(UnhashableMetric(), id="unhashable-callable")],
)
def test_cdist_diag_sim_falls_back_to_scipy(monkeypatch, metric):
    """Unsupported built-in and callable metrics retain the SciPy fallback."""
    x_sample = np.array([[0.0, 1.0], [2.0, 4.0]])
    y_sample = np.array([[1.0, 1.0], [3.0, 1.0]])
    expected = 1 / (1 + np.diag(scipy_cdist(x_sample, y_sample, metric=metric)))
    calls = []

    def spy_cdist(x, y, metric):
        calls.append(metric)
        return scipy_cdist(x, y, metric=metric)

    monkeypatch.setattr(compute, "cdist", spy_cdist)
    actual = compute._cdist_diag_sim(x_sample, y_sample, metric)

    np.testing.assert_allclose(actual, expected)
    assert calls == [metric]


def test_get_similarity_fn_dispatches_scipy_metric_fallback(monkeypatch):
    """The public API sends non-specialized SciPy metrics through cdist."""
    feats = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 2.0]])
    pairs = np.array([[0, 1], [1, 2]])
    expected = 1 / (
        1 + np.diag(scipy_cdist(feats[pairs[:, 0]], feats[pairs[:, 1]], "canberra"))
    )
    calls = []

    def spy_cdist(x, y, metric):
        calls.append(metric)
        return scipy_cdist(x, y, metric=metric)

    monkeypatch.setattr(compute, "cdist", spy_cdist)
    similarity = compute.get_similarity_fn("canberra", progress_bar=False)(
        feats, pairs, batch_size=2
    )

    np.testing.assert_allclose(similarity, expected)
    assert similarity.dtype == np.float32
    assert calls == ["canberra"]


def test_get_similarity_fn_dispatches_callable_directly(monkeypatch):
    """The public API uses custom pairwise callables without the cdist adapter."""
    feats = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 2.0]])
    pairs = np.array([[0, 1], [1, 2]])
    calls = []

    def pairwise_metric(x, y):
        calls.append((x.copy(), y.copy()))
        return np.sum(np.abs(x - y), axis=1)

    def fail_cdist_adapter(*args, **kwargs):
        raise AssertionError("custom callable unexpectedly used _cdist_diag_sim")

    monkeypatch.setattr(compute, "_cdist_diag_sim", fail_cdist_adapter)
    similarity = compute.get_similarity_fn(pairwise_metric, progress_bar=False)(
        feats, pairs, batch_size=2
    )

    np.testing.assert_allclose(similarity, [3.0, 4.0])
    assert len(calls) == 1


def test_null_dist_cached():
    """Test that null_dist_cached creates and uses cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        null_dist = compute.null_dist_cached(
            num_pos=5, total=20, seed=42, null_size=100, cache_dir=cache_dir
        )

        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)
        assert np.all(null_dist <= 1)

        # Cache stores one .npy file per (total, num_pos) pair
        npy_files = list(cache_dir.glob("*.npy"))
        assert len(npy_files) == 1
        assert npy_files[0].name == "n20_k5.npy"


def test_null_dist_cached_hit():
    """Test that null_dist_cached returns consistent results on cache hit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        kwargs = dict(num_pos=5, total=20, seed=42, null_size=100, cache_dir=cache_dir)

        first = compute.null_dist_cached(**kwargs)
        second = compute.null_dist_cached(**kwargs)
        assert np.array_equal(first, second)


def test_null_dist_cached_corrupt():
    """Test that a corrupted cache file is regenerated, not propagated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = cache_dir / "n20_k5.npy"

        # Write garbage to simulate corruption
        path.write_bytes(b"not a valid npy file")

        null_dist = compute.null_dist_cached(
            num_pos=5, total=20, seed=42, null_size=100, cache_dir=cache_dir
        )
        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)
        assert np.all(null_dist <= 1)


def _parallel_worker(args):
    """Worker for parallel cache stress test."""
    cache_dir, num_pos, total, seed, null_size = args
    return compute.null_dist_cached(num_pos, total, seed, null_size, Path(cache_dir))


def test_null_dist_cached_parallel():
    """Test that parallel workers don't corrupt the cache.

    Spawns 16 workers all racing on the same cache key.
    """
    from multiprocessing import Pool

    num_pos, total, seed, null_size = 5, 100, 42, 10_000
    n_workers = 16

    with tempfile.TemporaryDirectory() as tmpdir:
        args = [(tmpdir, num_pos, total, seed, null_size)] * n_workers

        with Pool(n_workers) as pool:
            results = pool.map(_parallel_worker, args)

        # All workers should return identical results
        for r in results:
            assert np.array_equal(results[0], r)
