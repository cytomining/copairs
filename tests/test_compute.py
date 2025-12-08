"""Test pairwise distance calculation functions."""

import tempfile
from math import comb
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

        # Generate null distribution with caching
        null_dist = compute.null_dist_cached(
            num_pos=5, total=20, seed=42, null_size=100, cache_dir=cache_dir
        )

        # Check it created a valid distribution
        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)
        assert np.all(null_dist <= 1)


def test_null_dist_cached_corrupt():
    """Test that null_dist_cached handles corrupted cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Create a corrupted cache file
        cache_file = cache_dir / "n20_k5.npy"
        cache_file.write_text("corrupted data")

        # Should regenerate despite corruption
        with pytest.warns(UserWarning, match="Failed to load cache file"):
            null_dist = compute.null_dist_cached(
                num_pos=5, total=20, seed=42, null_size=100, cache_dir=cache_dir
            )

        # Check it generated a valid distribution
        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)
        assert np.all(null_dist <= 1)


def test_exact_ap():
    """Test exact_ap computes AP for all possible rankings."""
    num_pos = 3
    total = 6
    n_combinations = comb(total, num_pos)  # 20 combinations

    exact_dist = compute.exact_ap(num_pos, total)

    # Should have exactly C(total, num_pos) AP scores
    assert len(exact_dist) == n_combinations
    assert np.all(exact_dist >= 0)
    assert np.all(exact_dist <= 1)

    # Best case: all positives at the start (positions 0, 1, 2)
    # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
    assert np.max(exact_dist) == pytest.approx(1.0)

    # Worst case: all positives at the end (positions 3, 4, 5)
    # AP = (1/4 + 2/5 + 3/6) / 3 = (0.25 + 0.4 + 0.5) / 3 ≈ 0.383
    assert np.min(exact_dist) == pytest.approx((1 / 4 + 2 / 5 + 3 / 6) / 3)


def test_exact_null_dist_used_when_small():
    """Test that exact computation is used when combinations < null_size."""
    num_pos = 3
    total = 6
    null_size = 100
    n_combinations = comb(total, num_pos)  # 20 < 100

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        null_dist = compute.null_dist_cached(
            num_pos=num_pos,
            total=total,
            seed=42,
            null_size=null_size,
            cache_dir=cache_dir,
        )

        # Should be padded to null_size
        assert len(null_dist) == null_size

        # Should contain exactly n_combinations unique values (the exact distribution)
        unique_values = np.unique(null_dist)
        assert len(unique_values) == n_combinations


def test_random_null_dist_used_when_large():
    """Test that random sampling is used when combinations > null_size."""
    num_pos = 10
    total = 100
    null_size = 1000
    # comb(100, 10) = 17,310,309,456,440 >> 1000

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        null_dist = compute.null_dist_cached(
            num_pos=num_pos,
            total=total,
            seed=42,
            null_size=null_size,
            cache_dir=cache_dir,
        )

        # Should have null_size samples
        assert len(null_dist) == null_size

        # With random sampling, we expect many unique values (not exactly n_combinations)
        unique_values = np.unique(null_dist)
        # Random sampling won't produce exactly 1000 unique values due to collisions
        # but should be reasonably close
        assert len(unique_values) > 500
