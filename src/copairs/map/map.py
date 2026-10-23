"""Functions to compute mean average precision."""

import logging
from os import cpu_count
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from copairs import nulls, compute
from copairs.nulls.common import check_null_size
from copairs.nulls.backend import check_method
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
    method: str = "fast",
    backend: str = "auto",
) -> pd.DataFrame:
    """Compute mAP scores and p-values from AP scores.

    This function groups AP scores by the specified columns, computes the mean
    Average Precision (mAP) for each group, and calculates p-values by comparing
    against null distributions.

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
        Location to save the cache (``method="legacy"`` only).
    method : str
        Both methods average the members' AP nulls, members with equal pair
        counts sharing samples, as copairs 0.5.5 did. ``"fast"`` (default)
        samples them exactly and streams them in bounded memory, without a
        cache (:func:`copairs.nulls.map_pvalues`), and counts group nulls
        ``>= mAP``, treating values within the scores' rounding as ties (see
        :func:`copairs.nulls.tie_thresholds`). ``"legacy"`` materializes the
        nulls of :func:`copairs.compute.get_null_dists` and counts group nulls
        ``> mAP``, as copairs 0.5.5 did, which gives too small p-values when
        the mAP equals an atom of the null (e.g. perfect retrieval).
    backend : str
        Backend of the fast method: ``"auto"`` (default), ``"numba"`` or
        ``"numpy"``. All give identical p-values, so ``"auto"`` picks
        ``"numba"`` whenever it is installed.

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
    check_method(method, backend)
    # Filter out invalid or incomplete AP scores
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    ap_scores = ap_scores.reset_index(drop=True).copy()

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
    map_scores["p_value"] = _copied_pvalues(
        map_scores["mean_average_precision"].to_numpy(),
        map_scores["indices"].tolist(),
        ap_scores,
        null_size=null_size,
        seed=seed,
        cache_dir=cache_dir,
        progress_bar=progress_bar,
        max_workers=max_workers,
        method=method,
        backend=backend,
    )
    return map_scores


def _copied_pvalues(
    scores,
    groups,
    ap_scores,
    null_size,
    seed,
    cache_dir,
    progress_bar,
    max_workers,
    method="fast",
    backend="auto",
):
    """P-values against the mean of the members' AP nulls, as in copairs <= 0.5.5.

    ``groups`` lists each group's rows of ``ap_scores``; members with equal pair
    counts share null samples. ``method="fast"`` streams the exact nulls of
    :func:`nulls.map_pvalues` and counts ties; ``"legacy"`` averages the nulls
    of :func:`compute.get_null_dists` and counts null mAPs ``>`` the score.
    """
    if not groups:
        check_null_size(null_size)
        nulls.resolve_backend(backend)
        return np.zeros(0, dtype=np.float64)
    members = np.concatenate(groups).astype(np.int64)
    rows = np.unique(members)
    null_confs = ap_scores.loc[rows, ["n_pos_pairs", "n_total_pairs"]].values
    null_confs, conf_ix = np.unique(null_confs, axis=0, return_inverse=True)
    rev_ix = np.full(len(ap_scores), -1, dtype=np.int64)
    rev_ix[rows] = conf_ix.ravel()

    if method == "fast":
        logger.info("Computing p-values...")
        gid = np.repeat(np.arange(len(groups)), [len(ix) for ix in groups])
        keys, conf_cnt = np.unique(
            gid * len(null_confs) + rev_ix[members], return_counts=True
        )
        ptr = np.searchsorted(keys // len(null_confs), np.arange(len(groups) + 1))
        return nulls.map_pvalues(
            scores,
            ptr,
            keys % len(null_confs),
            conf_cnt,
            null_confs,
            null_size,
            seed,
            backend=backend,
            progress_bar=progress_bar,
        )

    logger.info("Computing null_dist...")
    null_dists = compute.get_null_dists(
        null_confs,
        null_size,
        seed=seed,
        cache_dir=cache_dir,
        progress_bar=progress_bar,
        method="legacy",
    )

    def get_p_value(k):
        null_dist = null_dists[rev_ix[groups[k]]].mean(axis=0)
        # A Python float, as copairs 0.5.5 compared: float32 nulls stay float32.
        num = (null_dist > float(scores[k])).sum()
        return (num + 1) / (null_size + 1)  # Add 1 for stability

    logger.info("Computing p-values...")
    return _thread_map(get_p_value, range(len(groups)), progress_bar, max_workers)


def _thread_map(fn, items, progress_bar, max_workers):
    """``[fn(item) for item in items]`` on a thread pool."""
    if progress_bar:
        from tqdm.contrib.concurrent import thread_map

        return thread_map(fn, items, leave=False, max_workers=max_workers)
    return silent_thread_map(fn, items, max_workers=max_workers)


def mean_average_precision(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    method: str = "fast",
    backend: str = "auto",
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
        Location to save the cache (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"`` null, see :func:`get_map_pvalue`.
    backend : str
        Backend of the fast method, see :func:`get_map_pvalue`.

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
        method=method,
        backend=backend,
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
    method: str = "fast",
    backend: str = "auto",
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
        Location to save the cache (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"`` null, see :func:`get_map_pvalue`.
    backend : str
        Backend of the fast method, see :func:`get_map_pvalue`.

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
        method=method,
        backend=backend,
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
