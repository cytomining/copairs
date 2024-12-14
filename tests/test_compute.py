"""Test pairwise distance calculation functions."""

import numpy as np

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


def test_corrcoef():
    """Test correlation coefficient computation."""
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    corr_gt = corrcoef_naive(feats, pairs)
    corr_fn = compute.get_distance_fn("correlation")
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
    cosine_fn = compute.get_distance_fn("cosine")
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
    euclidean_fn = compute.get_distance_fn("euclidean")
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
    abs_cosine_fn = compute.get_distance_fn("abs_cosine")
    abs_cosine = abs_cosine_fn(feats, pairs, batch_size)
    assert np.allclose(abs_cosine_gt, abs_cosine)
