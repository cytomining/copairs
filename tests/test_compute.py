"""Test pairwise distance calculation functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from copairs import compute

SEED = 0
rng = np.random.default_rng(SEED)


def test_resolve_max_workers(monkeypatch):
    """Worker resolution is bounded, configurable, and task-aware."""
    monkeypatch.delenv("COPAIRS_MAX_WORKERS", raising=False)
    monkeypatch.setattr(compute.os, "cpu_count", lambda: 384)
    monkeypatch.setattr(
        compute.os, "sched_getaffinity", lambda _: set(range(64)), raising=False
    )
    assert compute._resolve_max_workers(1000, None) == 8
    assert compute._resolve_max_workers(3, None) == 3
    assert compute._resolve_max_workers(1000, 7) == 7

    monkeypatch.setenv("COPAIRS_MAX_WORKERS", "5")
    assert compute._resolve_max_workers(1000, None) == 5
    assert compute._resolve_max_workers(2, None) == 2
    assert compute._resolve_max_workers(0, None) == 0
    assert compute._resolve_max_workers(1000, 4) == 4


def test_resolve_max_workers_uses_affinity_with_fallback(monkeypatch):
    """Defaults respect process affinity and portably fall back to CPU count."""
    monkeypatch.delenv("COPAIRS_MAX_WORKERS", raising=False)
    monkeypatch.setattr(
        compute.os, "sched_getaffinity", lambda _: set(range(4)), raising=False
    )
    monkeypatch.setattr(compute.os, "cpu_count", lambda: 384)
    assert compute._resolve_max_workers(100, None) == 4

    def unavailable(_):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(compute.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(compute.os, "cpu_count", lambda: 6)
    assert compute._resolve_max_workers(100, None) == 6


@pytest.mark.parametrize("num_items", [0, 2])
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_resolve_max_workers_rejects_invalid_environment(monkeypatch, num_items, value):
    """The process-wide worker override is validated even without tasks."""
    monkeypatch.setenv("COPAIRS_MAX_WORKERS", value)
    with pytest.raises(ValueError, match="positive integer"):
        compute._resolve_max_workers(num_items, None)


@pytest.mark.parametrize("num_items", [0, 2])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_resolve_max_workers_rejects_invalid_argument(num_items, value):
    """Explicit worker budgets are validated even without tasks."""
    error = TypeError if isinstance(value, (float, bool)) else ValueError
    with pytest.raises(error, match="positive integer"):
        compute._resolve_max_workers(num_items, value)


def test_parallel_map_empty_validates_worker_budget(monkeypatch):
    """Empty task collections do not bypass worker-budget validation."""
    with pytest.raises(ValueError, match="positive integer"):
        compute.parallel_map(lambda _: None, [], progress_bar=False, max_workers=0)

    monkeypatch.setenv("COPAIRS_MAX_WORKERS", "invalid")
    with pytest.raises(ValueError, match="positive integer"):
        compute.parallel_map(lambda _: None, [], progress_bar=False)


def test_parallel_map_serial_honors_progress_bar(monkeypatch):
    """Serial execution still reports task progress."""
    progress_calls = []

    def fake_tqdm(tasks, **kwargs):
        progress_calls.append(kwargs)
        return tasks

    monkeypatch.setattr("tqdm.autonotebook.tqdm", fake_tqdm)
    monkeypatch.setattr(
        compute,
        "ThreadPool",
        lambda *_: pytest.fail("serial execution created a thread pool"),
    )
    visited = []
    compute.parallel_map(
        visited.append,
        np.arange(3),
        progress_bar=True,
        max_workers=1,
    )

    assert visited == [0, 1, 2]
    assert progress_calls == [{"total": 3, "leave": False}]


def test_batched_similarity_output_parity_across_worker_budgets():
    """Worker count does not affect batched similarity output."""
    feats = rng.normal(size=(12, 7))
    pairs = rng.integers(0, len(feats), size=(31, 2))
    similarity_fn = compute.get_similarity_fn("cosine", progress_bar=False)

    serial = similarity_fn(feats, pairs, 4, max_workers=1)
    parallel = similarity_fn(feats, pairs, 4, max_workers=4)

    np.testing.assert_array_equal(serial, parallel)


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


def test_null_distribution_output_parity_across_worker_budgets(tmp_path):
    """Native null-distribution output is unchanged by the worker budget."""
    confs = np.asarray([[1, 5], [2, 7], [3, 9]])
    kwargs = {
        "confs": confs,
        "null_size": 100,
        "seed": 42,
        "progress_bar": False,
    }

    serial = compute.get_null_dists(
        **kwargs, cache_dir=tmp_path / "serial", max_workers=1
    )
    parallel = compute.get_null_dists(
        **kwargs, cache_dir=tmp_path / "parallel", max_workers=3
    )

    np.testing.assert_array_equal(serial, parallel)


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
