"""Functions to compute mean average precision."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs import compute

logger = logging.getLogger("copairs")


def mean_average_precision(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int,
    progress_bar: Optional[bool] = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
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
    progress_bar : bool
        Whether or not to show tqdm's progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - `mean_average_precision`: Mean AP score for each group.
        - `p_value`: p-value comparing mAP to the null distribution.
        - `corrected_p_value`: Adjusted p-value after multiple testing correction.
        - `below_p`: Boolean indicating if the p-value is below the threshold.
        - `below_corrected_p`: Boolean indicating if the corrected p-value is below the threshold.
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
        }
    )
    map_scores.columns = sameby + ["mean_average_precision", "indices"]

    # Compute p-values for each group using the null distributions
    params = map_scores[["mean_average_precision", "indices"]]

    if progress_bar:
        from tqdm.contrib.concurrent import thread_map
    else:
        thread_map = silent_thread_map

    map_scores["p_value"] = thread_map(
        get_p_value, params.values, leave=False, max_workers=max_workers
    )

    # Perform multiple testing correction on p-values
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        map_scores["p_value"], method="fdr_bh"
    )
    map_scores["corrected_p_value"] = pvals_corrected

    # Mark scores below the p-value threshold
    map_scores["below_p"] = map_scores["p_value"] < threshold
    map_scores["below_corrected_p"] = map_scores["corrected_p_value"] < threshold

    return map_scores


def silent_thread_map(fn, *iterables, **kwargs):
    """Map iterables and kwargs to a function.

    Parameters
    ----------
    max_workers  : [default: min(32, cpu_count() + 4)].
    chunksize  : [default: 1].
    """
    # Based on tqdm's original implementation for consistency
    # (github.com/tqdm/tqdm/blob/0ed5d7f18fa3153834cbac0aa57e8092b217cc16/tqdm/contrib/concurrent.py#L29).

    kwargs = kwargs.copy()
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    chunksize = kwargs.pop("chunksize", 1)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, *iterables, chunksize=chunksize, **kwargs))
