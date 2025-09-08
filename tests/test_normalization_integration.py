"""Integration tests for normalized AP in the pipeline."""

import numpy as np
import pandas as pd
import pytest


def test_average_precision_with_normalization():
    """Test that average_precision can compute normalized scores."""
    from copairs.map.average_precision import average_precision

    # Create synthetic test data
    np.random.seed(42)
    n_profiles = 100
    n_features = 50

    # Create metadata with treatment groups
    treatments = ["compound_A"] * 20 + ["compound_B"] * 20 + ["DMSO"] * 60
    meta = pd.DataFrame(
        {
            "treatment": treatments,
            "batch": np.random.choice(["batch1", "batch2"], n_profiles),
            "well": [f"well_{i}" for i in range(n_profiles)],
        }
    )

    # Create feature matrix with some signal
    feats = np.random.randn(n_profiles, n_features)
    # Add signal for compound_A profiles to be similar
    feats[:20] += np.random.randn(1, n_features) * 0.5
    # Add signal for compound_B profiles to be similar
    feats[20:40] += np.random.randn(1, n_features) * 0.5

    # Test that normalized AP is always computed
    result = average_precision(
        meta=meta,
        feats=feats,
        pos_sameby=["treatment"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=["treatment"],
    )

    assert "average_precision" in result.columns
    assert "normalized_average_precision" in result.columns

    # Check that normalized values are in expected range
    valid_mask = ~result["normalized_average_precision"].isna()
    normalized_values = result.loc[valid_mask, "normalized_average_precision"]
    assert np.all(normalized_values >= -1.0)
    assert np.all(normalized_values <= 1.0)


def test_normalized_ap_with_different_prevalences():
    """Test that normalization makes scores comparable across different prevalences."""
    from copairs.map.average_precision import average_precision

    np.random.seed(123)

    # Scenario 1: Low prevalence (few positives)
    meta1 = pd.DataFrame(
        {"treatment": ["compound_X"] * 5 + ["DMSO"] * 95, "id": range(100)}
    )
    feats1 = np.random.randn(100, 20)
    feats1[:5] += np.random.randn(1, 20) * 0.8  # Add signal

    result1 = average_precision(
        meta=meta1,
        feats=feats1,
        pos_sameby=["treatment"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=["treatment"],
    )

    # Scenario 2: High prevalence (many positives)
    meta2 = pd.DataFrame(
        {"treatment": ["compound_Y"] * 50 + ["DMSO"] * 50, "id": range(100)}
    )
    feats2 = np.random.randn(100, 20)
    feats2[:50] += np.random.randn(1, 20) * 0.8  # Similar signal strength

    result2 = average_precision(
        meta=meta2,
        feats=feats2,
        pos_sameby=["treatment"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=["treatment"],
    )

    # Raw AP scores will be very different due to prevalence
    ap1 = result1[result1["treatment"] == "compound_X"]["average_precision"].mean()
    ap2 = result2[result2["treatment"] == "compound_Y"]["average_precision"].mean()

    # But normalized scores should be more comparable
    norm1 = result1[result1["treatment"] == "compound_X"][
        "normalized_average_precision"
    ].mean()
    norm2 = result2[result2["treatment"] == "compound_Y"][
        "normalized_average_precision"
    ].mean()

    # With similar signal strength, normalized scores should be closer than raw scores
    raw_diff = abs(ap2 - ap1)
    norm_diff = abs(norm2 - norm1)

    # Normalization should reduce the prevalence-induced difference
    # The effect may be modest with random data, so we use a lenient threshold
    assert norm_diff <= raw_diff, (
        "Normalization should not increase prevalence-based differences"
    )


def test_mean_average_precision_with_normalization():
    """Test that mean_average_precision handles normalized scores."""
    from copairs.map.map import mean_average_precision

    # Create mock AP scores with normalization
    ap_scores = pd.DataFrame(
        {
            "treatment": ["A", "A", "B", "B", "C", "C"],
            "average_precision": [0.8, 0.7, 0.6, 0.5, 0.3, 0.2],
            "normalized_average_precision": [0.6, 0.5, 0.4, 0.3, 0.1, 0.0],
            "n_pos_pairs": [10, 10, 8, 8, 5, 5],
            "n_total_pairs": [50, 50, 40, 40, 30, 30],
        }
    )

    # Test that normalized mAP is always computed
    result = mean_average_precision(
        ap_scores=ap_scores,
        sameby=["treatment"],
        null_size=100,
        threshold=0.05,
        seed=42,
    )

    assert "mean_average_precision" in result.columns
    assert "mean_normalized_average_precision" in result.columns

    # Check that we get one row per treatment
    assert len(result) == 3
    assert set(result["treatment"]) == {"A", "B", "C"}

    # Check that normalized mAP values match expected calculations
    for treatment in ["A", "B", "C"]:
        mask = ap_scores["treatment"] == treatment
        expected_norm_map = ap_scores.loc[mask, "normalized_average_precision"].mean()
        actual_norm_map = result.loc[
            result["treatment"] == treatment, "mean_normalized_average_precision"
        ].values[0]
        assert abs(actual_norm_map - expected_norm_map) < 0.001
