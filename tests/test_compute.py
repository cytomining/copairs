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

        # Cache stores one .npy file per (total, num_pos, seed) triple
        npy_files = list(cache_dir.glob("*.npy"))
        assert len(npy_files) == 1
        assert npy_files[0].name == "n20_k5_s42.npy"


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
        path = cache_dir / "n20_k5_s42.npy"

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


def test_get_null_dists_empty(tmp_path):
    """No configurations give an empty array of null distributions."""
    confs = np.empty((0, 2), dtype=int)
    null_dists = compute.get_null_dists(
        confs, 100, seed=0, cache_dir=tmp_path, progress_bar=False
    )
    assert null_dists.shape == (0, 100)


def test_p_values_empty(tmp_path, monkeypatch):
    """No AP scores give no p-values."""
    # p_values has no cache_dir argument; keep the default cache out of ~/.copairs
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pvals = compute.p_values(
        np.array([]), np.empty((0, 2), dtype=int), 100, 0, progress_bar=False
    )
    assert len(pvals) == 0


def test_get_null_dists_unseeded_skips_cache(tmp_path):
    """Unseeded nulls are fresh draws and are never written to the cache."""
    confs = np.array([[3, 20], [5, 30]])
    kwargs = dict(null_size=200, seed=None, cache_dir=tmp_path, progress_bar=False)
    first = compute.get_null_dists(confs, **kwargs)
    second = compute.get_null_dists(confs, **kwargs)
    assert list(tmp_path.rglob("*.npy")) == []
    assert not np.array_equal(first, second)


def explicit_map_null(members, edges, null_size, seed):
    """Draw every similarity explicitly and sort each full list, as a reference for `random_map`."""
    rng = np.random.default_rng(seed)
    shared = rng.random((null_size, len(edges)))
    total = np.zeros(null_size)
    for i, (num_pos, num_neg) in enumerate(members):
        mine = [e for e, pair in enumerate(edges) if i in pair]
        private = rng.random((null_size, num_pos - len(mine) + num_neg))
        vals = np.concatenate([shared[:, mine], private], axis=1)
        labels = np.r_[np.ones(num_pos), np.zeros(num_neg)]
        rel = labels[np.argsort(-vals, axis=1)]
        precision = np.cumsum(rel, axis=1) / np.arange(1, num_pos + num_neg + 1)
        total += (precision * rel).sum(axis=1) / num_pos
    return total / len(members)


def test_random_map_singleton_is_the_ap_null():
    """A group of one member follows the same law as `random_ap`."""
    from scipy.stats import ks_2samp

    joint = compute.random_map([(3, 100)], [], 50_000, seed=1)
    single = compute.random_ap(50_000, 3, 103, seed=2).astype(np.float64)
    assert ks_2samp(joint, single).pvalue > 0.001


@pytest.mark.parametrize(
    "members, edges",
    [
        # four replicates: every pair inside the group is a positive
        ([(3, 100)] * 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        # a cycle of four, each member with one more positive outside the group
        ([(3, 50), (3, 60), (3, 50), (3, 60)], [(0, 1), (1, 2), (2, 3), (3, 0)]),
        # a path of four with no positives outside the group
        ([(1, 30), (2, 40), (2, 40), (1, 30)], [(0, 1), (1, 2), (2, 3)]),
    ],
)
def test_random_map_matches_explicit_simulation(members, edges):
    """Placing negatives by gaps reproduces drawing every similarity explicitly."""
    from scipy.stats import ks_2samp

    fast = compute.random_map(members, edges, 50_000, seed=3)
    slow = explicit_map_null(members, edges, 50_000, seed=4)
    assert ks_2samp(fast, slow).pvalue > 0.001


def test_random_map_narrows_with_replicates():
    """The mAP null of four replicates is tighter than one AP, wider than four independent."""
    members = [(3, 100)] * 4
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    null = compute.random_map(members, edges, 100_000, seed=5)
    sd_ap = compute.random_ap(100_000, 3, 103, seed=5).std()
    assert sd_ap / 2 < null.std() < sd_ap
    # Closed forms for 3 positives among 103 (shntnu/expected_ap, ap_moments.py):
    # E[AP] = 0.069263 and, with the pair (i, j) shared by both lists, sd(mAP) = 0.049071.
    assert abs(null.mean() - 0.069263) < 0.0007
    assert abs(null.std() - 0.049071) < 0.0007


def test_random_map_is_reproducible_and_validates():
    """Same inputs give the same stream; inconsistent inputs are rejected."""
    members, edges = [(2, 10), (2, 10), (2, 10)], [(0, 1), (1, 2), (0, 2)]
    a = compute.random_map(members, edges, 1000, seed=7)
    assert np.array_equal(a, compute.random_map(members, edges, 1000, seed=7))
    assert not np.array_equal(a, compute.random_map(members, edges, 1000, seed=8))
    with pytest.raises(ValueError):
        compute.random_map([(1, 10), (1, 10), (1, 10)], edges, 10, seed=7)
    with pytest.raises(ValueError):
        compute.random_map(members, [(0, 0)], 10, seed=7)
    with pytest.raises(ValueError, match="chunk"):
        compute.random_map(members, edges, 10, seed=7, chunk=0)
