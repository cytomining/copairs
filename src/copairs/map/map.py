"""Functions to compute mean average precision."""

import logging
import warnings
from os import cpu_count
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from copairs import compute
from copairs.map.hierarchical_fdr import (
    apply_fdr_correction,
    check_hierarchical_by,
    apply_hierarchical_fdr_correction,
)

logger = logging.getLogger("copairs")


def get_map_pvalue(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    seed: int,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    pos_pairs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute mAP scores and p-values from AP scores.

    This function groups AP scores by the specified columns, computes the mean
    Average Precision (mAP) for each group, and calculates p-values by comparing
    against a null distribution sampled jointly over the group's members
    (`compute.random_map`): every positive pair inside the group is one random
    number shared by both members' rank lists, so the null has the right width
    for the number of replicates.

    Parameters
    ----------
    ap_scores : pd.DataFrame
        DataFrame containing individual Average Precision (AP) scores and pair statistics
        (e.g., number of positive pairs `n_pos_pairs` and total pairs `n_total_pairs`).
    sameby : list or str
        Metadata column(s) used to group profiles for mAP calculation.
    null_size : int
        Number of samples in the null distribution for significance testing.
    seed : int
        Random seed for reproducibility.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    max_workers : int
        Number of workers used. Default defined by tqdm's `thread_map`.
    cache_dir : str or Path
        Location to save the cache (only used by the legacy fallback below).
    pos_pairs : np.ndarray, optional
        The `(n_pairs, 2)` positive pairs (row indices of `ap_scores`) used to
        compute the AP scores, as returned by
        `find_pairs(meta, sameby=pos_sameby, diffby=pos_diffby)` on the `meta`
        given to `average_precision`; `ap_scores` must keep the index that
        `average_precision` returned. When omitted, each group's positives are
        assumed to be its other members, which is the replicate-retrieval case
        (`n_pos_pairs == len(group) - 1`). Groups where that does not hold fall
        back to copairs' previous null with a warning.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - Columns from `sameby` (group identifiers).
        - `mean_average_precision`: Mean AP score for each group.
        - `mean_normalized_average_precision`: Mean normalized AP score (scale-independent).
        - `p_value`: p-value comparing mAP to the null distribution.
        - `indices`: List of indices in the original ap_scores for this group.

    """
    # Filter out invalid or incomplete AP scores, remembering the original rows
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    profile = ap_scores.index.to_numpy()
    ap_scores = ap_scores.reset_index(drop=True).copy()
    num_pos = ap_scores["n_pos_pairs"].to_numpy().astype(np.int64)
    num_neg = ap_scores["n_total_pairs"].to_numpy().astype(np.int64) - num_pos

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
    groups = map_scores["indices"].tolist()

    # Positive pairs inside each group, as local (i, j) index pairs
    if pos_pairs is None:
        edges_of = [
            [(i, j) for i in range(len(ix)) for j in range(i + 1, len(ix))]
            if (num_pos[ix] == len(ix) - 1).all()
            else None
            for ix in groups
        ]
    else:
        pos_pairs = np.asarray(pos_pairs)
        group_of = np.full(int(max(pos_pairs.max(), profile.max())) + 1, -1)
        local = np.empty_like(group_of)
        for g, ix in enumerate(groups):
            group_of[profile[ix]] = g
            local[profile[ix]] = np.arange(len(ix))
        ga, gb = group_of[pos_pairs[:, 0]], group_of[pos_pairs[:, 1]]
        inside = (ga >= 0) & (ga == gb)
        edges_of = [[] for _ in groups]
        for g, a, b in zip(
            ga[inside], local[pos_pairs[inside, 0]], local[pos_pairs[inside, 1]]
        ):
            edges_of[g].append((int(a), int(b)))

    signatures = [
        None
        if edges is None
        else (tuple(zip(num_pos[ix], num_neg[ix])), tuple(sorted(edges)))
        for ix, edges in zip(groups, edges_of)
    ]
    logger.info("Computing null_dist...")
    unique = sorted(set(sig for sig in signatures if sig is not None))
    null_of = {}

    def par_func(k):
        members, edges = unique[k]
        null_of[unique[k]] = np.sort(
            compute.random_map(members, edges, null_size, seed)
        )

    if unique:
        compute.parallel_map(par_func, np.arange(len(unique)), progress_bar)

    legacy = [i for i, sig in enumerate(signatures) if sig is None]
    if legacy:
        warnings.warn(
            f"{len(legacy)} group(s) have positives that are not their other members; "
            "pass pos_pairs to sample their mAP null jointly. Falling back to the "
            "previous null, which copies one draw across members with equal pair counts.",
            stacklevel=2,
        )
        null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
        null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
        null_dists = compute.get_null_dists(
            null_confs,
            null_size,
            seed=seed,
            cache_dir=cache_dir,
            progress_bar=progress_bar,
        )

    # Function to calculate the p-value for a mAP score based on the null distribution
    def get_p_value(k):
        map_score, indices = map_scores["mean_average_precision"].iat[k], groups[k]
        if signatures[k] is None:
            null_dist = null_dists[rev_ix[indices]].mean(axis=0)
            num = (null_dist > map_score).sum()
        else:
            null_dist = null_of[signatures[k]]
            num = null_size - np.searchsorted(null_dist, map_score - 1e-9, side="left")
        return (num + 1) / (null_size + 1)  # Add 1 for stability

    logger.info("Computing p-values...")
    if progress_bar:
        from tqdm.contrib.concurrent import thread_map

        p_values = thread_map(
            get_p_value, range(len(groups)), leave=False, max_workers=max_workers
        )
    else:
        p_values = silent_thread_map(
            get_p_value, range(len(groups)), max_workers=max_workers
        )
    map_scores["p_value"] = p_values

    return map_scores


def mean_average_precision(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    pos_pairs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Calculate the Mean Average Precision (mAP) score and associated p-values.

    This function computes the Mean Average Precision (mAP) score by grouping profiles
    based on the specified criteria (`sameby`). It calculates the significance of mAP
    scores by comparing them to a null distribution and performs multiple testing
    corrections using Benjamini-Hochberg FDR.

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

    See Also
    --------
    mean_average_precision_hierarchical : For hierarchical FDR correction with grouped data.

    """
    # Step 1: Compute mAP scores and p-values
    map_scores = get_map_pvalue(
        ap_scores=ap_scores,
        sameby=sameby,
        null_size=null_size,
        seed=seed,
        progress_bar=progress_bar,
        max_workers=max_workers,
        cache_dir=cache_dir,
        pos_pairs=pos_pairs,
    )

    # Step 2: Apply multiple testing correction
    map_scores = apply_fdr_correction(map_scores)

    # Step 3: Mark scores below the p-value threshold
    map_scores["below_p"] = map_scores["p_value"] < threshold
    map_scores["below_corrected_p"] = map_scores["corrected_p_value"] < threshold

    return map_scores


def mean_average_precision_hierarchical(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int,
    hierarchical_by: List[str],
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Calculate the Mean Average Precision (mAP) score with hierarchical FDR correction.

    This function computes the Mean Average Precision (mAP) score by grouping profiles
    based on the specified criteria (`sameby`). It applies hierarchical FDR correction
    appropriate for grouped hypothesis testing, such as dose-response data.

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
    hierarchical_by : list
        Metadata column(s) for hierarchical FDR correction. Enables two-stage testing:

        - Stage 1: Use minimum p-value within each group defined by `hierarchical_by`,
          then apply BH correction at the group level. A group passes if any member
          is significant.
        - Stage 2: For groups that pass Stage 1, apply BH correction to the
          individual tests within each group.

        This is designed for dose-response data where only high doses are expected
        to be active. The `hierarchical_by` columns must be a proper subset of `sameby`.
        For example, with `sameby=['compound', 'dose']` and `hierarchical_by=['compound']`,
        mAP is calculated per compound×dose, but FDR correction accounts for the
        grouped structure.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    max_workers : int
        Number of workers used. Default defined by tqdm's `thread_map`.
    cache_dir : str or Path
        Location to save the cache.

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
        - `stage1_p_value`: Group-level p-value from Stage 1 (minimum p-value).
        - `stage1_corrected_p_value`: BH-corrected Stage 1 p-value.
        - `stage1_significant`: Whether the group passed Stage 1.

    See Also
    --------
    mean_average_precision : For standard BH FDR correction.

    """
    check_hierarchical_by(hierarchical_by, sameby)
    # Step 1: Compute mAP scores and p-values
    map_scores = get_map_pvalue(
        ap_scores=ap_scores,
        sameby=sameby,
        null_size=null_size,
        seed=seed,
        progress_bar=progress_bar,
        max_workers=max_workers,
        cache_dir=cache_dir,
    )

    # Step 2: Apply hierarchical multiple testing correction
    # Includes stage1_* columns for transparency. Could drop these in future
    # for cleaner output (only corrected_p_value is needed downstream).
    map_scores = apply_hierarchical_fdr_correction(map_scores, hierarchical_by, sameby)

    # Step 3: Mark scores below the p-value threshold
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
    max_workers = kwargs.pop("max_workers", min(32, (cpu_count() or 1) + 4))
    chunksize = kwargs.pop("chunksize", 1)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, *iterables, chunksize=chunksize, **kwargs))
