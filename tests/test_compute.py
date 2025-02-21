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
