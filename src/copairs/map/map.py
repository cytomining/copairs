"""Functions to compute mean average precision."""

import logging
import warnings
from os import cpu_count
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from copairs import nulls, compute
from copairs.nulls import joint, permutation
from copairs.methods import check_null, check_method
from copairs.map.hierarchical_fdr import (
    apply_fdr_correction,
    check_hierarchical_by,
    apply_hierarchical_fdr_correction,
)
from copairs.map.average_precision import (
    RankPairs,
    attached_pairs,
    check_recomputable,
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
    pos_pairs: Optional[np.ndarray] = None,
    null: str | None = None,
    pairs: RankPairs | None = None,
) -> pd.DataFrame:
    """Compute mAP scores and p-values from AP scores.

    This function groups AP scores by the specified columns, computes the mean
    Average Precision (mAP) for each group, and calculates p-values by comparing
    against the null distribution of the group's mAP chosen by `null`. Under
    ``method="fast"`` a null mAP counts against the group when it is ``>=``
    the group's mAP, so an mAP equal to an atom of the null (e.g. perfect
    retrieval) counts the atom; values within the scores' rounding count as
    ties (`copairs.nulls.tie_thresholds`).

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
        Threads that run the permutation null's groups (default defined by
        `concurrent.futures.ThreadPoolExecutor`), compute the legacy copied
        null's p-values (default defined by tqdm's `thread_map`), or run the
        fast copied null's NumPy backend (default: the CPUs this process may
        run on). The Numba backend's kernels run on Numba's threads, set by
        ``NUMBA_NUM_THREADS`` or ``numba.set_num_threads``; under the
        permutation null the ``max_workers`` threads prepare groups while one
        kernel runs at a time. P-values do not depend on the threads.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples
        (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) offers every null, defaults to
        ``null="permutation"`` and counts ties as above; its copied null
        samples the members' AP nulls exactly and streams them in bounded
        memory, without a cache (:func:`copairs.nulls.map_pvalues`).
        ``"legacy"`` reproduces copairs <= 0.5.5, with each configuration's
        null seeded by its pair counts: the copied null only, averaging the
        cached nulls of :func:`copairs.compute.get_null_dists` and counting
        group nulls ``> mAP``, which gives too small p-values when the mAP
        equals an atom of the null (e.g. perfect retrieval). Other nulls
        raise under it.
    backend : str
        Backend of the fast method's nulls: ``"auto"`` (default), ``"numba"``
        or ``"numpy"``. ``"numba"`` compiles the scalar functions that define
        each null's draws (the copied null's AP samples, the joint null's
        samples, the permutation null's relabellings and their mAPs), and
        ``"numpy"`` computes the same draws vectorized. The two give
        bitwise-identical draws and p-values (tested), so ``"auto"`` picks
        ``"numba"`` whenever it is installed and imports. Ignored by
        ``method="legacy"``.
    pos_pairs : np.ndarray, optional
        The `(n_pairs, 2)` positive pairs used to compute the AP scores,
        indexing the rows of the frame `average_precision` returned (its
        `return_pairs=True` gives them, as `pairs.pos`); `ap_scores` must keep
        that frame's index. Pairs whose counts disagree with `n_pos_pairs`
        raise. When omitted, they are taken from ``pairs`` or, failing that,
        from the pairs `average_precision` attached to ``ap_scores``. Frames
        without either assume each group's positives are its other members,
        which is the replicate-retrieval case
        (`n_pos_pairs == len(group) - 1`), and a group of one profile gets
        that profile's AP null. Other groups fall back to the copied null with
        a warning. Joint null only.
    null : str, optional
        Null of the group mAP; by default the method's: ``"permutation"``
        under ``method="fast"``, ``"copied"`` under ``method="legacy"``.
        ``"permutation"`` relabels profiles
        (`copairs.nulls.permutation`): in each block of members ranked against
        the same references (e.g. a plate's replicates and controls), any of
        the block's profiles could have carried the group's label, and each
        draw recomputes the group's mAP from the observed similarities. Each
        group's positives must be its other members (for groups of one
        profile with positives in other groups, use ``"joint"``: there it is
        that profile's AP null), and its smallest p-value is about one over
        its number of distinct relabellings. It recomputes similarities, so
        APs computed with a distance that estimates its scale per batch
        (``"seuclidean"``, ``"mahalanobis"``) raise. ``"joint"`` samples
        the members' APs together (`copairs.nulls.joint`): a positive pair
        inside the group is one random number in both members' rank lists,
        and other similarities are independent, which real profiles are not
        (members share references, and some profiles are close to many
        others), so it can give too small p-values for groups of many
        replicates. ``"copied"`` is the null of copairs <= 0.5.5 (the only
        null of ``method="legacy"``): each
        member's AP null is sampled per `(n_pos_pairs, n_total_pairs)`
        configuration and the group's null averages them, so members with
        equal pair counts share one draw and the null does not narrow as
        replicates are added.
    pairs : RankPairs, optional
        The pairs the AP scores were computed with, as
        ``average_precision(..., return_pairs=True)`` returns them; their
        indices are rows of that call's frame, so ``ap_scores`` must keep its
        index. The permutation null needs them and, when not given, takes
        them from ``ap_scores.attrs["copairs_pairs"]``, where
        `average_precision` attaches them; frames that lost them (built,
        reloaded, pickled, or concatenated from several AP calls) need
        ``pairs=`` or another null. The joint null takes its positive pairs
        from ``pairs`` too, or from the attached pairs (see ``pos_pairs``).

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
    null = check_null(method, null)
    if null != "joint" and pos_pairs is not None:
        raise ValueError("pos_pairs is used by the joint null only")
    if null != "copied" and pos_pairs is None and pairs is None:
        pairs = attached_pairs(ap_scores)
    if null == "joint" and pos_pairs is None and pairs is not None:
        pos_pairs = pairs.pos
    if null == "permutation" and pairs is None:
        raise ValueError(
            "null='permutation' (the fast method's default) needs the pairs "
            "the APs were computed from. average_precision attaches them to "
            "the frame it returns, and these AP scores have none: they were "
            "built, loaded from disk, pickled or concatenated from several AP "
            "calls, or come from multilabel.average_precision, which this null "
            "does not support. Recompute the APs, pass pairs=, or choose "
            "null='joint' or null='copied'."
        )
    if null == "permutation":
        check_recomputable(pairs.similarity)

    if null == "permutation" or pos_pairs is not None:
        _check_row_index(ap_scores.index)

    # Filter out invalid or incomplete AP scores, remembering the original rows
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    profile = ap_scores.index.to_numpy()
    ap_scores = ap_scores.reset_index(drop=True).copy()

    # Average the APs in float64 and round the means to the APs' dtype: ties
    # carry the APs' rounding, and pandas averages float32 APs to float32 or
    # float64 depending on its version and the group keys. The reported mAP
    # is then the score the nulls count ties against, under any pandas.
    ap_dtype = ap_scores["average_precision"].to_numpy().dtype
    score_dtype = ap_dtype if ap_dtype.kind == "f" else np.dtype(np.float64)
    ap_scores["average_precision"] = ap_scores["average_precision"].astype(np.float64)

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
    map_scores["mean_average_precision"] = map_scores["mean_average_precision"].astype(
        score_dtype
    )
    groups = map_scores["indices"].tolist()
    scores = map_scores["mean_average_precision"].to_numpy()
    copied = dict(
        null_size=null_size,
        seed=seed,
        cache_dir=cache_dir,
        progress_bar=progress_bar,
        max_workers=max_workers,
        method=method,
        backend=backend,
    )

    if null == "copied":
        map_scores["p_value"] = _copied_pvalues(scores, groups, ap_scores, **copied)
        return map_scores

    if null == "permutation":
        logger.info("Computing p-values...")
        map_scores["p_value"] = permutation.permutation_pvalues(
            groups,
            profile,
            ap_scores["n_pos_pairs"].to_numpy(),
            ap_scores["n_total_pairs"].to_numpy(),
            pairs.pos,
            pairs.neg,
            pairs.similarity,
            null_size,
            seed,
            observed=scores,
            progress_bar=progress_bar,
            max_workers=max_workers,
            pos_diffby=pairs.pos_diffby,
            backend=backend,
        )
        return map_scores

    num_pos = ap_scores["n_pos_pairs"].to_numpy().astype(np.int64)
    num_neg = ap_scores["n_total_pairs"].to_numpy().astype(np.int64) - num_pos
    logger.info("Computing p-values...")
    p_values, fallback = joint.group_pvalues(
        scores, groups, num_pos, num_neg, null_size, seed, profile, pos_pairs,
        progress_bar=progress_bar, backend=backend,
    )  # fmt: skip
    if fallback:
        warnings.warn(
            f"{len(fallback)} group(s) have positives that are not their other "
            "members; pass pos_pairs to sample their mAP null jointly. Falling "
            "back to the copied null, which copies one draw across members with "
            "equal pair counts.",
            stacklevel=2,
        )
        p_values[fallback] = _copied_pvalues(
            scores[fallback], [groups[k] for k in fallback], ap_scores, **copied
        )
    map_scores["p_value"] = p_values
    return map_scores


def _check_row_index(index: pd.Index) -> None:
    """Raise unless ``index`` holds row numbers, which AP pairs can index."""
    if index.dtype.kind not in "iu" or (len(index) and index.min() < 0):
        raise ValueError(
            "the AP scores' index must hold the row numbers of the frame "
            "average_precision returned, which the pairs index; got an index "
            f"of dtype {index.dtype}"
            + ("" if index.dtype.kind not in "iu" else " with negative labels")
            + ". Keep that frame's index (no set_index or reset_index)."
        )


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
    members = np.concatenate(groups) if groups else np.zeros(0)
    members = members.astype(np.int64)
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
            max_workers=max_workers,
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
    pos_pairs: Optional[np.ndarray] = None,
    null: str | None = None,
    pairs: RankPairs | None = None,
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
        Number of threads, see `get_map_pvalue`.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples
        (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"``, see `get_map_pvalue`.
    backend : str
        Backend of the fast method's nulls, see `get_map_pvalue`.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores, for ``null="joint"``;
        see `get_map_pvalue`. Needed there when a group's positives are not
        its other members.
    null : str, optional
        Null of the group mAP: ``"permutation"`` (the fast method's default),
        ``"joint"`` or ``"copied"`` (the legacy method's); see
        `get_map_pvalue`.
    pairs : RankPairs, optional
        Pairs of the AP scores, needed by ``null="permutation"`` and attached
        to AP frames; see `get_map_pvalue`.

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
        pos_pairs=pos_pairs,
        null=null,
        pairs=pairs,
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
    pos_pairs: Optional[np.ndarray] = None,
    null: str | None = None,
    pairs: RankPairs | None = None,
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
        Number of threads, see `get_map_pvalue`.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples
        (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"``, see `get_map_pvalue`.
    backend : str
        Backend of the fast method's nulls, see `get_map_pvalue`.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores, for ``null="joint"``;
        see `get_map_pvalue`. Needed there when a group's positives are not
        its other members.
    null : str, optional
        Null of the group mAP: ``"permutation"`` (the fast method's default),
        ``"joint"`` or ``"copied"`` (the legacy method's); see
        `get_map_pvalue`.
    pairs : RankPairs, optional
        Pairs of the AP scores, needed by ``null="permutation"`` and attached
        to AP frames; see `get_map_pvalue`.

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
        pos_pairs=pos_pairs,
        null=null,
        pairs=pairs,
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
