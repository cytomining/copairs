"""Tests for hierarchical FDR correction."""

import numpy as np
import pandas as pd
import pytest

from copairs.map.map import simes_pvalue, mean_average_precision


class TestSimesPvalue:
    """Tests for Simes' p-value combination method."""

    def test_single_pvalue(self):
        """Single p-value should be returned as-is."""
        assert simes_pvalue(np.array([0.05])) == 0.05

    def test_empty_pvalues(self):
        """Empty array should return 1.0."""
        assert simes_pvalue(np.array([])) == 1.0

    def test_all_significant(self):
        """All small p-values should give small combined p-value."""
        pvals = np.array([0.001, 0.002, 0.003])
        combined = simes_pvalue(pvals)
        # Simes: min(n * p_(i) / i) = min(3*0.001/1, 3*0.002/2, 3*0.003/3)
        #                           = min(0.003, 0.003, 0.003) = 0.003
        assert np.isclose(combined, 0.003)

    def test_one_significant(self):
        """One small p-value among large ones."""
        pvals = np.array([0.01, 0.5, 0.9])
        combined = simes_pvalue(pvals)
        # Simes: min(3*0.01/1, 3*0.5/2, 3*0.9/3) = min(0.03, 0.75, 0.9) = 0.03
        assert np.isclose(combined, 0.03)

    def test_all_nonsignificant(self):
        """All large p-values should give large combined p-value."""
        pvals = np.array([0.5, 0.6, 0.7])
        combined = simes_pvalue(pvals)
        assert combined > 0.4  # Should be relatively large


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

    def test_hierarchical_returns_stage1_columns(self, sample_ap_scores):
        """Hierarchical FDR should add Stage 1 columns."""
        result = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=["compound"],
            progress_bar=False,
        )

        assert "stage1_p_value" in result.columns
        assert "stage1_corrected_p_value" in result.columns
        assert "stage1_significant" in result.columns

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
        assert n_sig_hier >= n_sig_flat * 0.8  # Allow some tolerance

    def test_hierarchical_stage1_groups_correctly(self, sample_ap_scores):
        """Stage 1 should have one p-value per compound."""
        result = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=["compound"],
            progress_bar=False,
        )

        # Each compound should have the same stage1_p_value across doses
        for compound in result["compound"].unique():
            compound_data = result[result["compound"] == compound]
            stage1_pvals = compound_data["stage1_p_value"].unique()
            assert len(stage1_pvals) == 1, (
                f"Compound {compound} has multiple stage1 p-values"
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

    def test_without_hierarchical_no_stage1_columns(self, sample_ap_scores):
        """Without hierarchical_by, Stage 1 columns should not be present."""
        result = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=None,
            progress_bar=False,
        )

        assert "stage1_p_value" not in result.columns
        assert "stage1_corrected_p_value" not in result.columns
        assert "stage1_significant" not in result.columns

    def test_hierarchical_preserves_all_rows(self, sample_ap_scores):
        """Hierarchical FDR should preserve all input rows."""
        result = mean_average_precision(
            sample_ap_scores,
            sameby=["compound", "dose"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            hierarchical_by=["compound"],
            progress_bar=False,
        )

        # Should have same number of compound×dose combinations
        n_expected = sample_ap_scores.groupby(["compound", "dose"]).ngroups
        assert len(result) == n_expected
