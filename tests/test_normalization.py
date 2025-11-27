"""Tests for AP normalization functions."""

import numpy as np
import pytest

from copairs.map.normalization import expected_ap, normalize_ap


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
