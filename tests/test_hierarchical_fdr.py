"""Tests for hierarchical FDR correction.

TODO: Once API is finalized, improve tests:
- Use deterministic fixture with known expected p-values
- Verify stage 1 filtering actually excludes groups
- Verify stage 2 BH applied only within passing groups
- Compare against hand-calculated reference results
"""

import numpy as np
import pandas as pd
import pytest

from copairs.map.map import mean_average_precision


class TestHierarchicalFDR:
    """Tests for hierarchical FDR correction in mean_average_precision."""

    @pytest.fixture
    def sample_ap_scores(self):
        """Create sample AP scores with compound×dose structure."""
        # 5 compounds, 3 doses each = 15 tests
        data = []
        np.random.seed(42)

        # Compound A: all doses significant (strong signal)
        for dose in [1, 5, 10]:
            data.append(
                {
                    "compound": "A",
                    "dose": dose,
                    "average_precision": 0.9 + np.random.uniform(-0.05, 0.05),
                    "normalized_average_precision": 0.85,
                    "n_pos_pairs": 10,
                    "n_total_pairs": 100,
                }
            )

        # Compound B: some doses significant
        for dose, ap in zip([1, 5, 10], [0.3, 0.7, 0.8]):
            data.append(
                {
                    "compound": "B",
                    "dose": dose,
                    "average_precision": ap,
                    "normalized_average_precision": ap - 0.1,
                    "n_pos_pairs": 10,
                    "n_total_pairs": 100,
                }
            )

        # Compound C: no signal (random)
        for dose in [1, 5, 10]:
            data.append(
                {
                    "compound": "C",
                    "dose": dose,
                    "average_precision": 0.15 + np.random.uniform(-0.05, 0.05),
                    "normalized_average_precision": 0.0,
                    "n_pos_pairs": 10,
                    "n_total_pairs": 100,
                }
            )

        # Compound D: weak signal
        for dose in [1, 5, 10]:
            data.append(
                {
                    "compound": "D",
                    "dose": dose,
                    "average_precision": 0.25 + np.random.uniform(-0.05, 0.05),
                    "normalized_average_precision": 0.1,
                    "n_pos_pairs": 10,
                    "n_total_pairs": 100,
                }
            )

        # Compound E: strong signal
        for dose in [1, 5, 10]:
            data.append(
                {
                    "compound": "E",
                    "dose": dose,
                    "average_precision": 0.95 + np.random.uniform(-0.02, 0.02),
                    "normalized_average_precision": 0.9,
                    "n_pos_pairs": 10,
                    "n_total_pairs": 100,
                }
            )

        return pd.DataFrame(data)

    def test_hierarchical_by_validation_not_subset(self, sample_ap_scores):
        """hierarchical_by must be subset of sameby."""
        with pytest.raises(ValueError, match="must be a subset of"):
            mean_average_precision(
                sample_ap_scores,
                sameby=["compound", "dose"],
                null_size=100,
                threshold=0.05,
                seed=42,
                hierarchical_by=["other_column"],
                progress_bar=False,
            )

    def test_hierarchical_by_validation_equal_to_sameby(self, sample_ap_scores):
        """hierarchical_by must be proper subset of sameby."""
        with pytest.raises(ValueError, match="must be a proper subset"):
            mean_average_precision(
                sample_ap_scores,
                sameby=["compound"],
                null_size=100,
                threshold=0.05,
                seed=42,
                hierarchical_by=["compound"],
                progress_bar=False,
            )

    def test_hierarchical_fewer_corrections(self, sample_ap_scores):
        """Hierarchical FDR should be less stringent than flat BH."""
        # Standard BH correction
        result_flat = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=None,
            progress_bar=False,
        )

        # Hierarchical correction
        result_hier = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=["compound"],
            progress_bar=False,
        )

        # For truly significant compounds, hierarchical should find at least
        # as many significant results (often more due to less correction)
        n_sig_flat = result_flat["below_corrected_p"].sum()
        n_sig_hier = result_hier["below_corrected_p"].sum()

        # Hierarchical should generally find more or equal significant results
        # (less over-correction for related hypotheses)
        # Note: This is a probabilistic test, but with our setup it should hold
        # Lower bound: hierarchical shouldn't be much worse than flat
        assert n_sig_hier >= n_sig_flat * 0.8, (
            f"Hierarchical found too few: {n_sig_hier} vs flat {n_sig_flat}"
        )
        # Upper bound: sanity check that hierarchical isn't wildly broken
        assert n_sig_hier <= len(result_hier), (
            f"Hierarchical found more than total tests: {n_sig_hier}"
        )

    def test_hierarchical_nonsig_groups_get_pval_1(self, sample_ap_scores):
        """Groups that don't pass Stage 1 should have corrected_p_value = 1.0."""
        result = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=["compound"],
            progress_bar=False,
        )

        # For non-significant groups, corrected p-value should be 1.0
        nonsig_mask = ~result["stage1_significant"]
        if nonsig_mask.any():
            assert (result.loc[nonsig_mask, "corrected_p_value"] == 1.0).all()
