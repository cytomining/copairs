import pytest
import numpy as np

from copairs import compute_np
from copairs.compute import TF_ENABLED, TFP_ENABLED
if TF_ENABLED:
    from copairs import compute_tf

SEED = 0
rng = np.random.default_rng(SEED)


def corrcoef_naive(feats, pairs):
    corr = np.empty((len(pairs), ))
    for pos, (i, j) in enumerate(pairs):
        corr[pos] = np.corrcoef(feats[i], feats[j])[0, 1]
    return corr


def cosine_naive(feats, pairs):
    cosine = np.empty((len(pairs), ))
    for pos, (i, j) in enumerate(pairs):
        a, b = feats[i], feats[j]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine[pos] = a.dot(b) / (norm_a * norm_b)
    return cosine


def _test_corrcoef(backend):
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    corr_gt = corrcoef_naive(feats, pairs)
    corr = backend.pairwise_indexed(feats, pairs, backend.pairwise_corr,
                                    batch_size)
    assert np.allclose(corr_gt, corr)


def _test_cosine(backend):
    n_samples = 10
    n_pairs = 20
    n_feats = 5
    batch_size = 4
    feats = rng.uniform(0, 1, [n_samples, n_feats])
    pairs = rng.integers(0, n_samples - 1, [n_pairs, 2])

    cosine_gt = cosine_naive(feats, pairs)
    cosine = backend.pairwise_indexed(feats, pairs, backend.pairwise_cosine,
                                      batch_size)
    assert np.allclose(cosine_gt, cosine)


def test_corrcoef_np():
    _test_corrcoef(compute_np)


def test_cosine_np():
    _test_cosine(compute_np)


@pytest.mark.skipif(not TFP_ENABLED, reason="tensorflow_prob not installed")
def test_corrcoef_tf():
    _test_corrcoef(compute_tf)


@pytest.mark.skipif(not TF_ENABLED, reason="tensorflow_prob not installed")
def test_cosine_tf():
    _test_cosine(compute_tf)
