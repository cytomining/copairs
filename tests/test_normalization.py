"""Tests for AP normalization functions."""

import numpy as np
import pytest

import copairs.map.normalization as normalization_module
from copairs.map.normalization import expected_ap, normalize_ap


def _scalar_normalize_ap_reference(ap, M, N, eps=1e-10):
    """Reproduce the pre-vectorization implementation for differential tests."""
    is_scalar = np.isscalar(ap)
    ap = np.atleast_1d(ap)
    M = np.atleast_1d(M)
    N = np.atleast_1d(N)

    mu0 = np.zeros_like(ap, dtype=float)
    for i in range(len(ap)):
        m = M[i] if len(M) > 1 else M[0]
        n = N[i] if len(N) > 1 else N[0]
        mu0[i] = expected_ap(int(m), int(n))

    denominator = np.maximum(1 - mu0, eps)
    normalized = np.clip((ap - mu0) / denominator, -1.0, 1.0)
    return float(normalized[0]) if is_scalar else normalized


def test_expected_ap_basic_properties():
    """Test basic properties of expected AP."""
    from copairs.map.normalization import expected_ap

    # Edge case: all positive (M=L)
    assert expected_ap(5, 0) == 1.0

    # Edge case: all negative (M=0)
    assert expected_ap(0, 5) == 0.0

    # Edge case: single item
    assert expected_ap(1, 0) == 1.0
    assert expected_ap(0, 1) == 0.0

    # Property: E[AP] > prevalence for finite samples
    M, N = 10, 40
    L = M + N
    prevalence = M / L
    mu0 = expected_ap(M, N)
    assert mu0 > prevalence

    # Property: E[AP] approaches prevalence as L increases
    M_large, N_large = 100, 400
    mu0_large = expected_ap(M_large, N_large)
    prevalence_large = M_large / (M_large + N_large)
    bias_large = mu0_large - prevalence_large

    M_small, N_small = 10, 40
    mu0_small = expected_ap(M_small, N_small)
    prevalence_small = M_small / (M_small + N_small)
    bias_small = mu0_small - prevalence_small

    assert bias_large < bias_small  # Bias decreases with larger L

    # Known value from Bestgen (2015): M=2, N=3 => E[AP] = 0.5925
    assert abs(expected_ap(2, 3) - 0.5925) < 0.001


def test_normalize_ap_properties():
    """Test key properties of AP normalization."""
    # Property 1: Random performance => normalized AP = 0
    M, N = 20, 80
    mu0 = expected_ap(M, N)
    normalized = normalize_ap(mu0, M, N)
    assert abs(normalized) < 1e-6, "Random performance should normalize to ~0"

    # Property 2: Perfect performance => normalized AP = 1
    perfect_ap = 1.0
    normalized = normalize_ap(perfect_ap, M, N)
    assert abs(normalized - 1.0) < 1e-6, "Perfect performance should normalize to ~1"

    # Property 3: Worse than random => negative normalized AP
    worse_than_random = mu0 * 0.5  # Half of expected
    normalized = normalize_ap(worse_than_random, M, N)
    assert normalized < 0, "Worse than random should be negative"

    # Property 4: Scale independence - different prevalences
    # Same "effect size" should give similar normalized scores
    M1, N1 = 5, 95  # 5% prevalence
    M2, N2 = 50, 50  # 50% prevalence

    mu0_1 = expected_ap(M1, N1)
    mu0_2 = expected_ap(M2, N2)

    # Create AP scores that are 50% better than random
    ap1 = mu0_1 + 0.5 * (1 - mu0_1)
    ap2 = mu0_2 + 0.5 * (1 - mu0_2)

    norm1 = normalize_ap(ap1, M1, N1)
    norm2 = normalize_ap(ap2, M2, N2)

    assert abs(norm1 - norm2) < 0.01, (
        "Similar effect sizes should have similar normalized scores"
    )
    assert abs(norm1 - 0.5) < 0.01, "50% improvement should normalize to ~0.5"


@pytest.mark.parametrize(
    ("ap", "positives", "negatives"),
    [
        pytest.param(
            np.array([0.2, 0.4, 0.7, 0.9]),
            np.array([2, 2, 5, 2]),
            np.array([8, 8, 5, 8]),
            id="repeated-configurations",
        ),
        pytest.param(
            np.array([0.1, 0.3, 0.6, 0.95]),
            np.array([2, 3, 7, 11]),
            np.array([3, 17, 93, 989]),
            id="unique-configurations",
        ),
        pytest.param(
            np.array([np.nan, 0.0, 1.0, 0.5, 0.8]),
            np.array([0, 0, 1, 1, 400]),
            np.array([25, 1, 0, 99_999, 99_600]),
            id="zero-one-positive-large-total-and-nan",
        ),
        pytest.param(
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([3, 8, 18]),
            id="broadcast-positive-count",
        ),
        pytest.param(
            np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]),
            np.array([2, 4]),
            np.array([8, 6]),
            id="multidimensional-ap-row-broadcasting",
        ),
    ],
)
def test_normalize_ap_matches_scalar_reference(ap, positives, negatives):
    """Vectorized expectations retain the scalar implementation's results."""
    expected = _scalar_normalize_ap_reference(ap, positives, negatives)
    actual = normalize_ap(ap, positives, negatives)

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0, equal_nan=True)


def test_normalize_ap_scalar_matches_scalar_reference_with_custom_epsilon():
    """Scalar output and a non-default denominator epsilon remain unchanged."""
    expected = _scalar_normalize_ap_reference(0.5, 3, 0, eps=0.25)
    actual = normalize_ap(0.5, 3, 0, eps=0.25)

    assert isinstance(actual, float)
    assert actual == expected


def test_normalize_ap_empty_input_matches_scalar_reference():
    """Empty array inputs remain supported and retain their output dtype."""
    ap = np.array([], dtype=np.float32)
    counts = np.array([], dtype=int)

    expected = _scalar_normalize_ap_reference(ap, counts, counts)
    actual = normalize_ap(ap, counts, counts)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_normalize_ap_empty_multidimensional_input_matches_scalar_reference():
    """Empty AP matrices preserve their trailing dimensions."""
    ap = np.empty((0, 3), dtype=np.float32)
    counts = np.array([], dtype=int)

    expected = _scalar_normalize_ap_reference(ap, counts, counts)
    actual = normalize_ap(ap, counts, counts)

    assert actual.shape == (0, 3)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("positives", "negatives"),
    [
        pytest.param(
            np.array([[2], [3]]),
            np.array([8, 7]),
            id="multidimensional-positive-counts",
        ),
        pytest.param(
            np.array([2, 3]),
            np.array([[8], [7]]),
            id="multidimensional-negative-counts",
        ),
    ],
)
def test_normalize_ap_rejects_multidimensional_counts(positives, negatives):
    """Count arrays must unambiguously map one value to each AP row."""
    with pytest.raises(
        ValueError, match="M and N must be scalars or one-dimensional arrays"
    ):
        normalize_ap(np.array([[0.2, 0.4], [0.3, 0.5]]), positives, negatives)


def test_normalize_ap_deduplicates_harmonic_numbers(monkeypatch):
    """Repeated totals require only one harmonic-number calculation each."""
    calls = []
    original_harmonic_number = normalization_module.harmonic_number

    def record_harmonic_number(total):
        calls.append(total)
        return original_harmonic_number(total)

    monkeypatch.setattr(normalization_module, "harmonic_number", record_harmonic_number)
    normalize_ap(
        np.array([0.2, 0.3, 0.4, 0.5]),
        np.array([2, 3, 2, 4]),
        np.array([8, 7, 8, 16]),
    )

    assert calls == [10, 20]


def test_normalize_ap_vectorized():
    """Test that normalization works with array inputs."""
    from copairs.map.normalization import normalize_ap

    # Multiple AP scores with different configurations
    ap_scores = np.array([0.3, 0.5, 0.8])
    M_values = np.array([10, 20, 30])
    N_values = np.array([90, 80, 70])

    normalized = normalize_ap(ap_scores, M_values, N_values)

    assert isinstance(normalized, np.ndarray)
    assert len(normalized) == len(ap_scores)
    assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)

    # Test scalar input still returns scalar
    single_norm = normalize_ap(0.5, 10, 90)
    assert np.isscalar(single_norm)


def test_normalize_ap_edge_cases():
    """Test edge cases in normalization."""
    from copairs.map.normalization import normalize_ap

    # When M=L (all positive), mu0=1, denominator approaches 0
    # Should handle gracefully without division by zero
    M, N = 100, 0
    ap = 1.0  # Perfect score when all are positive
    normalized = normalize_ap(ap, M, N)
    assert not np.isnan(normalized)
    assert not np.isinf(normalized)

    # Very small M (rare positives)
    M, N = 1, 999
    ap = 0.5
    normalized = normalize_ap(ap, M, N)
    assert not np.isnan(normalized)
    assert -1.0 <= normalized <= 1.0


def test_normalization_interpretability():
    """Test that normalized scores are interpretable."""
    from copairs.map.normalization import expected_ap, normalize_ap

    M, N = 25, 75
    mu0 = expected_ap(M, N)

    # Create a range of AP scores
    ap_scores = np.linspace(0, 1, 11)
    normalized = normalize_ap(ap_scores, M, N)

    # Check monotonicity: higher AP => higher normalized AP
    assert np.all(np.diff(normalized) >= 0), "Normalization should preserve order"

    # Check specific interpretable points
    random_idx = np.argmin(np.abs(ap_scores - mu0))
    assert abs(normalized[random_idx]) < 0.1, (
        "Near-random performance should normalize near 0"
    )

    assert normalized[0] < 0, "AP=0 should give negative normalized score"
    assert normalized[-1] > 0.99, "AP=1 should give normalized score near 1"
