"""Hierarchical FDR correction for grouped hypothesis testing."""

import logging
from typing import List

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger("copairs")


def _multipletests(
    pvals: np.ndarray, method: str = "fdr_bh"
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Correct only tested p-values; retain missing results and unknown rejection."""
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    corrected = np.full(len(pvals), np.nan)
    reject = np.zeros(len(pvals), dtype=bool)
    sidak = bonf = np.nan
    if valid.any():
        reject[valid], corrected[valid], sidak, bonf = multipletests(
            pvals[valid], method=method
        )
    if not valid.all():
        reject = pd.array(reject, dtype="boolean")
        reject[~valid] = pd.NA
    return reject, corrected, sidak, bonf


def check_group_columns(columns, name, frame=None):
    """Return a nonempty list of distinct grouping column names, or raise."""
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, (list, tuple, pd.Index)):
        raise ValueError(
            f"{name} must be a column name or a nonempty list of column names"
        )
    columns = list(columns)
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError(f"{name} must contain nonempty string column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} must contain distinct column names")
    if frame is not None:
        for column in columns:
            if list(frame.columns).count(column) != 1:
                raise ValueError(
                    f"{name} column {column!r} must occur exactly once in the input"
                )
    return columns


def check_hierarchical_by(hierarchical_by: list[str], sameby: list[str]) -> None:
    """Raise ValueError unless ``hierarchical_by`` is a nonempty proper subset."""
    hierarchical_by = check_group_columns(hierarchical_by, "hierarchical_by")
    sameby = check_group_columns(sameby, "sameby")
    if not set(hierarchical_by).issubset(set(sameby)):
        raise ValueError(
            f"hierarchical_by columns {hierarchical_by} must be a subset of "
            f"sameby columns {sameby}"
        )

    if set(hierarchical_by) == set(sameby):
        raise ValueError(
            f"hierarchical_by columns {hierarchical_by} must be a proper subset of "
            f"sameby columns {sameby}. If they are equal, use standard correction "
            f"by not specifying hierarchical_by."
        )


def apply_hierarchical_fdr_correction(
    map_scores: pd.DataFrame,
    hierarchical_by: List[str],
    sameby: List[str],
) -> pd.DataFrame:
    """Apply hierarchical FDR correction for grouped hypotheses.

    Implements a two-stage testing procedure appropriate for dose-response data
    where only high doses are expected to be active:

    - Stage 1: Use minimum p-value within each group defined by `hierarchical_by`,
      then apply BH correction at the group level. A group passes if any member
      is significant.
    - Stage 2: For groups that pass Stage 1, apply BH correction to the
      individual tests within each group.

    Parameters
    ----------
    map_scores : pd.DataFrame
        DataFrame containing mAP scores with a 'p_value' column.
    hierarchical_by : list
        Metadata column(s) defining the group structure (e.g., ['compound']).
    sameby : list
        Metadata column(s) used for mAP calculation (e.g., ['compound', 'dose']).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - `corrected_p_value`: BH-corrected p-value (1.0 for groups that didn't pass Stage 1).
        - `stage1_p_value`: Group-level p-value from Stage 1 (minimum p-value).
        - `stage1_corrected_p_value`: BH-corrected Stage 1 p-value.
        - `stage1_significant`: Whether the group passed Stage 1.

    Raises
    ------
    ValueError
        If `hierarchical_by` is not a proper subset of `sameby`.

    Notes
    -----
    This method uses minimum p-value (rather than Simes) for Stage 1 aggregation.
    Min-p is appropriate for dose-response data where only high doses are expected
    to be active. Simes would penalize compounds for having inactive low doses,
    which is the expected biological behavior.

    """
    sameby = check_group_columns(sameby, "sameby", map_scores)
    hierarchical_by = check_group_columns(hierarchical_by, "hierarchical_by")
    check_hierarchical_by(hierarchical_by, sameby)

    logger.info("Applying hierarchical FDR correction...")
    map_scores = map_scores.copy()

    # Stage 1: Aggregate p-values to group level using minimum p-value
    # Min-p is appropriate for dose-response where only high doses are expected to be active
    stage1_pvals = map_scores.groupby(hierarchical_by, observed=True).agg(
        {"p_value": "min"}
    )
    stage1_pvals.columns = ["stage1_p_value"]

    # Apply BH correction at the group level
    reject_stage1, stage1_corrected, _, _ = _multipletests(
        stage1_pvals["stage1_p_value"], method="fdr_bh"
    )
    stage1_pvals["stage1_corrected_p_value"] = stage1_corrected
    stage1_pvals["stage1_significant"] = reject_stage1

    # Merge Stage 1 results back to map_scores
    map_scores = map_scores.merge(
        stage1_pvals.reset_index(), on=hierarchical_by, how="left"
    )

    # Stage 2: For groups that passed Stage 1, apply BH within each group
    # For groups that didn't pass, set corrected_p_value to 1.0
    map_scores["corrected_p_value"] = np.where(
        map_scores["p_value"].notna(), 1.0, np.nan
    )

    for group_key, group_df in map_scores.groupby(hierarchical_by, observed=True):
        passed = group_df["stage1_significant"].iloc[0]
        if pd.isna(passed) or not passed:
            # Group didn't pass Stage 1, skip
            continue

        group_indices = group_df.index
        group_pvals = group_df["p_value"].values

        if len(group_pvals) == 1:
            # Single test in group, no additional correction needed
            map_scores.loc[group_indices, "corrected_p_value"] = group_pvals[0]
        else:
            # Apply BH correction within the group
            _, group_corrected, _, _ = _multipletests(group_pvals, method="fdr_bh")
            map_scores.loc[group_indices, "corrected_p_value"] = group_corrected

    return map_scores


def apply_fdr_correction(
    map_scores: pd.DataFrame,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply standard FDR correction across all tests.

    Parameters
    ----------
    map_scores : pd.DataFrame
        DataFrame containing mAP scores with a 'p_value' column.
    method : str, optional
        Multiple testing correction method (default: 'fdr_bh').
        See statsmodels.stats.multitest.multipletests for options.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'corrected_p_value' column added.

    """
    map_scores = map_scores.copy()
    _, pvals_corrected, _, _ = _multipletests(map_scores["p_value"], method=method)
    map_scores["corrected_p_value"] = pvals_corrected
    return map_scores
