"""Tests for the fast null method in copairs' public p-value functions."""

import numpy as np
import pandas as pd
import pytest

from copairs import nulls, compute
from copairs.map import mean_average_precision
from tests.helpers import brute_ap_pvalues
from copairs.map.normalization import expected_ap

BACKENDS = nulls.available_backends()
CONFS = np.array([[1, 7], [3, 10], [5, 20], [9, 49], [99, 1099], [3, 1000]])


def test_legacy_method_is_unchanged():
    """method="legacy" reproduces copairs <= 0.5.5's per-configuration seeds."""
    confs = np.array([[2, 10], [3, 30]])
    got = compute.get_null_dists(
        confs, 100, seed=0, progress_bar=False, method="legacy"
    )
    seeds = np.random.default_rng(0).integers(8096, size=len(confs))
    for row, (num_pos, total), seed in zip(got, confs, seeds):
        np.testing.assert_array_equal(row, compute.random_ap(100, num_pos, total, seed))


def perfect_retrieval_scores():
    """Return two groups of two replicates (one positive each) among 100 profiles."""
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "average_precision": [1.0, 1.0, 0.5, 0.5],
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 100,
        }
    )


def test_map_pvalue_counts_ties(tmp_path):
    """An mAP equal to a null atom counts the atom.

    Under the copied null P(mAP >= 1) = P(AP >= 1) = 1/100. Under the joint
    null, given the shared similarity u each member's rank - 1 is
    Binomial(99, 1 - u): both APs are 1 with probability int u^198 = 1/199, and
    mAP >= 0.5 needs one rank of 1 or both of 2.
    """
    kwargs = dict(null_size=100_000, threshold=0.05, seed=0, progress_bar=False)
    both_two = 99**2 * 2 / (197 * 198 * 199)  # int (99 (1 - u) u^98)^2 du
    for null, expected in (
        ("copied", np.array([0.01, 0.02])),
        ("joint", np.array([1 / 199, 2 / 100 - 1 / 199 + both_two])),
    ):
        fast = mean_average_precision(
            perfect_retrieval_scores(), ["g"], null=null, **kwargs
        )
        se = np.sqrt(expected * (1 - expected) / kwargs["null_size"])
        assert (np.abs(fast["p_value"].to_numpy() - expected) < 5 * se).all()
    legacy = mean_average_precision(
        perfect_retrieval_scores(), ["g"], cache_dir=tmp_path, method="legacy", **kwargs
    )
    assert legacy["p_value"].iloc[0] == pytest.approx(1 / 100_001)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_scores_count_ties(backend):
    """float32-rounded scores still tie with the null atoms they round from.

    With one positive among 3, AP is 1, 1/2 or 1/3, so every null value is
    ``>= 1/3`` and the p-value of an AP or mAP of 1/3 is exactly 1.
    """
    kwargs = dict(null_size=1000, seed=0, progress_bar=False, backend=backend)
    for dtype in (np.float64, np.float32):
        p = compute.p_values(np.array([1 / 3], dtype=dtype), [[1, 3]], **kwargs)
        assert p[0] == 1.0
    scores = pd.DataFrame(
        {
            "g": ["a", "a"],
            "average_precision": np.float32([1 / 3, 1 / 3]),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 3,
        }
    )
    fast = mean_average_precision(scores, ["g"], threshold=0.05, **kwargs)
    assert fast["p_value"].iloc[0] == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_map_fast_vs_legacy(backend, tmp_path):
    """Fast copied-null and legacy mAP p-values agree within Monte Carlo error."""
    rng = np.random.default_rng(3)
    n = 400
    dframe = pd.DataFrame(
        {
            "g": rng.integers(60, size=n).astype(str),
            "n_pos_pairs": rng.integers(3, 8, size=n),
            "n_total_pairs": 0,
        }
    )
    dframe["n_total_pairs"] = dframe["n_pos_pairs"] + rng.integers(40, 60, size=n)
    m, total = dframe["n_pos_pairs"], dframe["n_total_pairs"]
    mu = np.array([expected_ap(a, b - a) for a, b in zip(m, total)])
    dframe["average_precision"] = np.clip(mu + rng.normal(0, 0.15, size=n), 0, 1)
    dframe["normalized_average_precision"] = 0.0
    kwargs = dict(null_size=20_000, threshold=0.05, seed=0, progress_bar=False)
    fast = mean_average_precision(
        dframe, ["g"], backend=backend, null="copied", **kwargs
    )
    legacy = mean_average_precision(
        dframe, ["g"], cache_dir=tmp_path, method="legacy", **kwargs
    )
    p_fast, p_legacy = fast["p_value"].to_numpy(), legacy["p_value"].to_numpy()
    se = np.sqrt(p_legacy * (1 - p_legacy) / 20_000 * 2) + 1e-4
    assert (np.abs(p_fast - p_legacy) < 6 * se).all()
    np.testing.assert_array_equal(
        fast["mean_average_precision"], legacy["mean_average_precision"]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_p_values_fast(backend):
    """compute.p_values with the fast method equals the brute-force count."""
    rng = np.random.default_rng(7)
    null_confs = CONFS[rng.integers(len(CONFS), size=200)]
    scores = rng.random(200)
    got = compute.p_values(
        scores, null_confs, 3000, seed=1, progress_bar=False, backend=backend
    )
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    expected = brute_ap_pvalues(scores, rev_ix.ravel(), confs, 3000, 1)
    np.testing.assert_array_equal(got, expected.astype(np.float32))


def test_unknown_method():
    """Unknown null methods raise."""
    with pytest.raises(ValueError):
        compute.get_null_dists(CONFS, 10, 0, method="exact")


@pytest.mark.parametrize("method", ["fast", "legacy"])
def test_unknown_backend(method, tmp_path):
    """A misspelled backend raises under either method, though legacy ignores it."""
    scores = np.full(len(CONFS), 0.5)
    with pytest.raises(ValueError, match="unknown backend"):
        compute.p_values(scores, CONFS, 10, 0, method=method, backend="cdua")
    with pytest.raises(ValueError, match="unknown backend"):
        compute.get_null_dists(
            CONFS, 10, 0, cache_dir=tmp_path, method=method, backend="cdua"
        )


def test_unknown_null():
    """Unknown nulls, and nulls legacy does not implement, raise."""
    scores = perfect_retrieval_scores()
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision(scores, ["g"], null="exact", **kwargs)
    with pytest.raises(ValueError, match="supports only null='copied'"):
        mean_average_precision(scores, ["g"], method="legacy", null="joint", **kwargs)


@pytest.mark.parametrize("null", ["joint", "copied"])
def test_map_fast_empty_input(null):
    """No valid AP scores give an empty result with the usual columns."""
    empty = pd.DataFrame(
        {
            "g": pd.Series([], dtype=str),
            "average_precision": pd.Series([], dtype=float),
            "normalized_average_precision": pd.Series([], dtype=float),
            "n_pos_pairs": pd.Series([], dtype=int),
            "n_total_pairs": pd.Series([], dtype=int),
        }
    )
    result = mean_average_precision(
        empty, ["g"], 100, 0.05, 0, progress_bar=False, null=null
    )
    assert len(result) == 0
    assert {"p_value", "corrected_p_value", "below_corrected_p"} <= set(result.columns)
