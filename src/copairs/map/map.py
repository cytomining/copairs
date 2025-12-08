"""Functions to compute mean average precision."""

import logging
from os import cpu_count
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs import compute

logger = logging.getLogger("copairs")


def simes_pvalue(pvalues: np.ndarray) -> float:
    """Combine p-values using Simes' method.

    Simes' method provides a combined p-value that is valid under independence
    and positive dependence (PRDS condition).

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values to combine.

    Returns
    -------
    float
        Combined p-value.
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return 1.0
    if n == 1:
        return float(pvalues[0])
    sorted_pvals = np.sort(pvalues)
    # Simes' formula: min(n * p_(i) / i) for i = 1, ..., n
    ranks = np.arange(1, n + 1)
    adjusted = n * sorted_pvals / ranks
    return float(np.min(adjusted))


def mean_average_precision(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    hierarchical_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate the Mean Average Precision (mAP) score and associated p-values.

    This function computes the Mean Average Precision (mAP) score by grouping profiles
    based on the specified criteria (`sameby`). It calculates the significance of mAP
    scores by comparing them to a null distribution and performs multiple testing
    corrections.

    Parameters
    ----------
    ap_scores : pd.DataFrame
        DataFrame containing individual Average Precision (AP) scores and pair statistics
        (e.g., number of positive pairs `n_pos_pairs` and total pairs `n_total_pairs`).
    sameby : list or str
        Metadata column(s) used to group profiles for mAP calculation.
    null_size : int
        Number of samples in the null distribution for significance testing.
    threshold : float
        p-value threshold for identifying significant MaP scores.
    seed : int
        Random seed for reproducibility.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    max_workers : int
        Number of workers used. Default defined by tqdm's `thread_map`.
    cache_dir : str or Path
        Location to save the cache.
    hierarchical_by : list, optional
        Metadata column(s) for hierarchical FDR correction. When specified, enables
        two-stage testing (Yekutieli 2008):

        - Stage 1: Aggregate p-values within each group defined by `hierarchical_by`
          using Simes' method, then apply BH correction at the group level.
        - Stage 2: For groups that pass Stage 1, apply BH correction to the
          individual tests within each group.

        This reduces over-correction when testing related hypotheses (e.g., multiple
        doses of the same compound). The `hierarchical_by` columns must be a subset
        of `sameby`. For example, with `sameby=['compound', 'dose']` and
        `hierarchical_by=['compound']`, mAP is calculated per compound×dose, but
        FDR correction accounts for the grouped structure.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - `mean_average_precision`: Mean AP score for each group.
        - `mean_normalized_average_precision`: Mean normalized AP score (scale-independent).
        - `p_value`: p-value comparing mAP to the null distribution.
        - `corrected_p_value`: Adjusted p-value after multiple testing correction.
        - `below_p`: Boolean indicating if the p-value is below the threshold.
        - `below_corrected_p`: Boolean indicating if the corrected p-value is below the threshold.

        When `hierarchical_by` is used, additional columns are included:
        - `stage1_p_value`: Group-level p-value from Stage 1 (Simes' aggregation).
        - `stage1_corrected_p_value`: BH-corrected Stage 1 p-value.
        - `stage1_significant`: Whether the group passed Stage 1.

    References
    ----------
    Yekutieli, D. (2008). "Hierarchical false discovery rate-controlling methodology."
    Journal of the American Statistical Association, 103(481):309-316.
    """
    # Filter out invalid or incomplete AP scores
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    ap_scores = ap_scores.reset_index(drop=True).copy()

    logger.info("Computing null_dist...")
    # Extract configurations for null distribution generation
    null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
    null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)

    # Generate null distributions for each unique configuration
    null_dists = compute.get_null_dists(
        null_confs, null_size, seed=seed, cache_dir=cache_dir, progress_bar=progress_bar
    )
    ap_scores["null_ix"] = rev_ix

    # Function to calculate the p-value for a mAP score based on the null distribution
    def get_p_value(params):
        map_score, indices = params
        null_dist = null_dists[rev_ix[indices]].mean(axis=0)
        num = (null_dist > map_score).sum()
        p_value = (num + 1) / (null_size + 1)  # Add 1 for stability
        return p_value

    logger.info("Computing p-values...")

    # Group by the specified metadata column(s) and calculate mean AP
    map_scores = ap_scores.groupby(sameby, observed=True, as_index=False).agg(
        {
            "average_precision": ["mean", lambda x: list(x.index)],
            "normalized_average_precision": "mean",
        }
    )
    map_scores.columns = sameby + [
        "mean_average_precision",
        "indices",
        "mean_normalized_average_precision",
    ]

    # Compute p-values for each group using the null distributions
    params = map_scores[["mean_average_precision", "indices"]]

    if progress_bar:
        from tqdm.contrib.concurrent import thread_map

        p_values = thread_map(
            get_p_value, params.values, leave=False, max_workers=max_workers
        )
    else:
        p_values = silent_thread_map(
            get_p_value, params.values, max_workers=max_workers
        )
    map_scores["p_value"] = p_values

    # Perform multiple testing correction on p-values
    if hierarchical_by is None:
        # Standard BH correction across all tests
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
            map_scores["p_value"], method="fdr_bh"
        )
        map_scores["corrected_p_value"] = pvals_corrected
    else:
        # Hierarchical FDR correction (Yekutieli 2008)
        # Validate that hierarchical_by is a subset of sameby
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

        logger.info("Applying hierarchical FDR correction...")

        # Stage 1: Aggregate p-values to group level using Simes' method
        stage1_pvals = map_scores.groupby(hierarchical_by, observed=True).agg(
            {"p_value": simes_pvalue}
        )
        stage1_pvals.columns = ["stage1_p_value"]

        # Apply BH correction at the group level
        reject_stage1, stage1_corrected, _, _ = multipletests(
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
        map_scores["corrected_p_value"] = 1.0

        for group_key, group_df in map_scores.groupby(hierarchical_by, observed=True):
            if not group_df["stage1_significant"].iloc[0]:
                # Group didn't pass Stage 1, skip
                continue

            group_indices = group_df.index
            group_pvals = group_df["p_value"].values

            if len(group_pvals) == 1:
                # Single test in group, no additional correction needed
                map_scores.loc[group_indices, "corrected_p_value"] = group_pvals[0]
            else:
                # Apply BH correction within the group
                _, group_corrected, _, _ = multipletests(group_pvals, method="fdr_bh")
                map_scores.loc[group_indices, "corrected_p_value"] = group_corrected

    # Mark scores below the p-value threshold
    map_scores["below_p"] = map_scores["p_value"] < threshold
    map_scores["below_corrected_p"] = map_scores["corrected_p_value"] < threshold

    return map_scores


def silent_thread_map(fn, *iterables, **kwargs):
    """Map iterables and kwargs to a function.

    Parameters
    ----------
    fn : callable
        Function to map over iterables.
    *iterables : tuple
        Iterables to map over.
    **kwargs : dict
        Additional keyword arguments. Accepts:
        - max_workers : int, optional
            Maximum number of workers [default: min(32, cpu_count() + 4)].
        - chunksize : int, optional
            Size of chunks for each worker [default: 1].
    """
    # Based on tqdm's original implementation for consistency
    # (github.com/tqdm/tqdm/blob/0ed5d7f18fa3153834cbac0aa57e8092b217cc16/tqdm/contrib/concurrent.py#L29).

    kwargs = kwargs.copy()
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    chunksize = kwargs.pop("chunksize", 1)
    kwargs.pop("leave", None)  # Remove tqdm-specific kwarg
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, *iterables, chunksize=chunksize, **kwargs))
