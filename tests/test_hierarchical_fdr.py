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

import copairs.map.map as map_module
from copairs.map.map import mean_average_precision, mean_average_precision_hierarchical
from copairs.matching import UnpairedException
from copairs.map.hierarchical_fdr import (
    apply_fdr_correction,
    apply_hierarchical_fdr_correction,
)


def run_map(ap_scores, hierarchical, cache_dir):
    """Run mAP per compound and dose, with or without hierarchical FDR."""
    kwargs = dict(
        sameby=["compound", "dose"],
        null_size=100,
        threshold=0.05,
        seed=42,
        progress_bar=False,
        cache_dir=cache_dir,
    )
    if hierarchical:
        return mean_average_precision_hierarchical(
            ap_scores, hierarchical_by=["compound"], **kwargs
        )
    return mean_average_precision(ap_scores, **kwargs)


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

    @pytest.mark.parametrize(
        "sameby, hierarchical_by, match",
        [
            (["compound", "dose"], ["other_column"], "must be a subset of"),
            (["compound"], ["compound"], "must be a proper subset"),
        ],
    )
    def test_hierarchical_by_validation(
        self, sample_ap_scores, monkeypatch, sameby, hierarchical_by, match
    ):
        """hierarchical_by must be a proper subset of sameby, checked before p-values."""
        monkeypatch.setattr(
            map_module,
            "get_map_pvalue",
            lambda *args, **kwargs: pytest.fail("p-values computed before validation"),
        )
        with pytest.raises(ValueError, match=match):
            mean_average_precision_hierarchical(
                sample_ap_scores,
                sameby=sameby,
                hierarchical_by=hierarchical_by,
                null_size=100,
                threshold=0.05,
                seed=42,
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
            progress_bar=False,
        )

        # Hierarchical correction
        result_hier = mean_average_precision_hierarchical(
            sample_ap_scores,
            sameby=["compound", "dose"],
            hierarchical_by=["compound"],
            null_size=1000,
            threshold=0.05,
            seed=42,
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
        result = mean_average_precision_hierarchical(
            sample_ap_scores,
            sameby=["compound", "dose"],
            hierarchical_by=["compound"],
            null_size=1000,
            threshold=0.05,
            seed=42,
            progress_bar=False,
        )

        # For non-significant groups, corrected p-value should be 1.0
        nonsig_mask = ~result["stage1_significant"]
        if nonsig_mask.any():
            assert (result.loc[nonsig_mask, "corrected_p_value"] == 1.0).all()

    @pytest.mark.parametrize("hierarchical", [False, True])
    @pytest.mark.parametrize(
        "invalidate",
        [
            lambda df: df.assign(average_precision=np.nan),
            lambda df: df.assign(n_pos_pairs=0),
            lambda df: df.iloc[:0],
        ],
        ids=["nan_ap", "no_positives", "no_rows"],
    )
    def test_no_valid_ap_scores(
        self, sample_ap_scores, tmp_path, invalidate, hierarchical
    ):
        """No valid AP scores raise UnpairedException in both mAP entry points."""
        with pytest.raises(UnpairedException, match="valid AP scores"):
            run_map(invalidate(sample_ap_scores), hierarchical, tmp_path)

    @pytest.mark.parametrize("hierarchical", [False, True])
    @pytest.mark.parametrize(
        "column, value", [("average_precision", np.nan), ("n_pos_pairs", 0)]
    )
    def test_one_valid_ap_score(
        self, sample_ap_scores, tmp_path, column, value, hierarchical
    ):
        """One valid AP score among invalid ones still gives its group's mAP."""
        ap_scores = sample_ap_scores.copy()
        ap_scores.loc[ap_scores.index[1:], column] = value
        result = run_map(ap_scores, hierarchical, tmp_path)

        assert len(result) == 1
        first_ap = ap_scores["average_precision"].iloc[0]
        assert result["mean_average_precision"].iloc[0] == first_ap

    def test_fdr_correction_no_tests(self):
        """FDR correction of zero p-values returns an empty table."""
        empty = pd.DataFrame(
            {
                "compound": pd.Series([], dtype=str),
                "dose": pd.Series([], dtype=int),
                "p_value": pd.Series([], dtype=float),
            }
        )
        standard = apply_fdr_correction(empty)
        hierarchical = apply_hierarchical_fdr_correction(
            empty, ["compound"], ["compound", "dose"]
        )
        for result in (standard, hierarchical):
            assert len(result) == 0
            assert "corrected_p_value" in result.columns
