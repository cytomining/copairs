"""Test pairwise distance calculation functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_cosine_pairs_exactly_match_generic_batches(dtype, layout):
    """Prepared pair dots match gathered cosine across dtypes and layouts."""
    local_rng = np.random.default_rng(SEED)
    feats = local_rng.normal(size=(17, 11)).astype(dtype)
    if layout == "fortran":
        feats = np.asfortranarray(feats)
    elif layout == "strided":
        backing = np.empty((len(feats), feats.shape[1] * 2), dtype=dtype)
        backing[:, ::2] = feats
        feats = backing[:, ::2]
    pairs = np.asarray(
        [(i, (i * 7 + 3) % len(feats)) for i in range(len(feats))],
        dtype=np.uint32,
    )
    generic = compute.get_similarity_fn("cosine", progress_bar=False)(
        feats, pairs, batch_size=3
    )
    normalized = compute.prepare_cosine(feats, np.unique(pairs))
    actual = compute.cosine_pairs(
        normalized, pairs, batch_size=3, progress_bar=False
    )

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, generic)


def test_prepare_cosine_skips_unreferenced_nonfinite_rows_under_strict_errstate():
    """Only pair-referenced profiles participate in normalization."""
    feats = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [np.inf, 1.0]]
    )
    pairs = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)

    with np.errstate(all="raise"):
        normalized = compute.prepare_cosine(feats, np.unique(pairs))
        actual = compute.cosine_pairs(
            normalized, pairs, batch_size=2, progress_bar=False
        )
        expected = compute.pairwise_cosine(
            feats[pairs[:, 0]], feats[pairs[:, 1]]
        ).astype(np.float32)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad_row", [[0.0, 0.0], [np.inf, 1.0]])
def test_prepare_cosine_preserves_referenced_nonfinite_errstate(bad_row):
    """Referenced zero and infinite rows retain strict NumPy error behavior."""
    feats = np.asarray([[1.0, 0.0], bad_row])

    with np.errstate(all="raise"), pytest.raises(FloatingPointError):
        compute.prepare_cosine(feats, np.asarray([1]))


def test_cosine_pairs_preserve_zero_norm_and_nonfinite_results():
    """Pre-normalization retains generic cosine behavior for zero-norm rows."""
    feats = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    pairs = np.asarray([[0, 1], [1, 2], [1, 1]], dtype=np.uint32)

    with np.errstate(invalid="ignore"):
        generic = compute.get_similarity_fn("cosine", progress_bar=False)(
            feats, pairs, batch_size=1
        )
        actual = compute.cosine_pairs(
            compute.prepare_cosine(feats, np.unique(pairs)),
            pairs,
            batch_size=2,
            progress_bar=False,
        )

    np.testing.assert_array_equal(actual, generic)
    assert np.isnan(actual[0])


def test_cosine_pairs_returns_typed_empty_result(monkeypatch):
    """An empty pair array returns without starting parallel workers."""

    def fail_parallel_map(*args, **kwargs):
        raise AssertionError("parallel_map must not run for empty pairs")

    monkeypatch.setattr(compute, "parallel_map", fail_parallel_map)
    actual = compute.cosine_pairs(
        np.empty((3, 2)),
        np.empty((0, 2), dtype=np.uint32),
        batch_size=2,
        progress_bar=False,
    )

    assert actual.shape == (0,)
    assert actual.dtype == np.float32


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
