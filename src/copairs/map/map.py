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
from copairs.methods import check_null, check_method
from copairs.map.hierarchical_fdr import (
    apply_fdr_correction,
    check_hierarchical_by,
    apply_hierarchical_fdr_correction,
)
from copairs.map.average_precision import RankPairs, LabelPairs, attached_pairs

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
    null: Optional[str] = None,
    pairs: RankPairs | None = None,
) -> pd.DataFrame:
    """Compute mAP scores and p-values from AP scores.

    This function groups AP scores by the specified columns, computes the mean
    Average Precision (mAP) for each group, and calculates p-values by comparing
    against null distributions. With the fast method, a group's null is the
    distribution of its mAP when its label is given to other profiles
    (:mod:`copairs.nulls.permutation`), recomputed from the observed
    similarities: the paper's null hypothesis that the group's profiles and its
    references come from the same distribution, tested without assuming
    anything about how similarities depend on each other.

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
        ``"fast"`` (default) streams exact null samples and counts group nulls
        ``>= mAP``, treating values within the scores' rounding as ties (see
        :func:`copairs.nulls.tie_thresholds`), in bounded memory. ``"legacy"`` reproduces copairs <= 0.5.5, which counts
        group nulls ``> mAP`` and so gives too small p-values when the observed
        mAP equals an atom of the null (e.g. perfect retrieval).
    backend : str
        Backend of the fast method: ``"auto"``, ``"cuda"``, ``"numba"`` or
        ``"numpy"``.
    pos_pairs : np.ndarray, optional
        The `(n_pairs, 2)` positive pairs used to compute the AP scores,
        indexing the rows of the frame `average_precision` returned (its
        `return_pairs=True` gives them, as `pairs.pos`); `ap_scores` must keep
        that frame's index. Pairs whose counts disagree with `n_pos_pairs`
        raise. When omitted, each group's positives are
        assumed to be its other members, which is the replicate-retrieval case
        (`n_pos_pairs == len(group) - 1`). Groups where that does not hold fall
        back to copairs' previous null with a warning. Joint null only; taken
        from ``pairs`` (single-label APs) when available.
    null : str, optional
        Null of the group mAP. ``"permutation"`` (the fast method's default)
        relabels profiles: in each block of members ranked against the same
        references (e.g. a plate's replicates and controls), any of the
        block's profiles could have carried the group's label; for multilabel
        APs, the group's label moves to other profiles of its components of
        the negative graph, which keep their other labels. Each group's
        positives must be its other members, and its smallest p-value is
        about one over its number of distinct relabellings. ``"joint"``
        samples the members' APs jointly (:mod:`copairs.nulls.joint`): a
        positive pair inside the group is one random number in both members'
        rank lists, and other similarities are independent, which real
        profiles are not (members share references, and some profiles are
        close to many others), so it can give too small p-values for groups
        of many replicates. ``"copied"`` averages the members' AP nulls with
        members of equal configuration sharing samples, as in copairs <= 0.5.5
        (the only null of ``method="legacy"``): it does not narrow as
        replicates are added, and members with different pair counts get
        independent samples.
    pairs : RankPairs or LabelPairs, optional
        The pairs and similarity the AP scores were computed with. Taken from
        ``ap_scores.attrs["copairs_pairs"]``, where
        :func:`~copairs.map.average_precision` and
        :func:`~copairs.map.multilabel.average_precision` attach them, when
        not given; frames that lost them (built, reloaded, pickled, or
        concatenated from several AP calls) need ``pairs=`` or another null.
        ``ap_scores`` must keep the index of the frame the pairs came with.

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
    if pairs is None:
        pairs = attached_pairs(ap_scores)
    if null != "joint" and pos_pairs is not None:
        raise ValueError("pos_pairs is used by the joint null only")
    if null == "permutation" and pairs is None:
        raise ValueError(
            "null='permutation' (the fast method's default) needs the pairs the "
            "APs were computed from. average_precision attaches them to the "
            "frame it returns, and these AP scores have none: they were built, "
            "loaded from disk, pickled or concatenated from several AP calls. "
            "Recompute the APs, pass pairs=, or choose null='joint' or "
            "null='copied'."
        )
    try:
        return _map_pvalues(
            ap_scores, sameby, null_size, seed, progress_bar, max_workers,
            cache_dir, method, backend, pos_pairs, null, pairs,
        )  # fmt: skip
    finally:
        release = getattr(getattr(pairs, "similarity", None), "release", None)
        if release is not None:
            release()


def _map_pvalues(
    ap_scores, sameby, null_size, seed, progress_bar, max_workers, cache_dir,
    method, backend, pos_pairs, null, pairs,
):  # fmt: skip
    """:func:`get_map_pvalue` with its null and pairs resolved."""
    # Filter out invalid or incomplete AP scores, remembering the original rows
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    profile = ap_scores.index.to_numpy()
    ap_scores = ap_scores.reset_index(drop=True).copy()

    if null == "joint" and pos_pairs is None and type(pairs) is RankPairs:
        pos_pairs = pairs.pos  # LabelPairs index profiles, not AP rows
    null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
    null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    rev_ix = rev_ix.ravel()

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

    if null == "permutation" and isinstance(pairs, LabelPairs):
        if pairs.label_col not in ap_scores:
            raise ValueError(
                f"null='permutation' of multilabel APs needs their {pairs.label_col!r} "
                "column"
            )
        logger.info("Computing p-values...")
        map_scores["p_value"] = nulls.permutation.label_permutation_pvalues(
            map_scores["indices"].tolist(),
            profile,
            ap_scores[pairs.label_col].to_numpy(),
            ap_scores["n_pos_pairs"].to_numpy(),
            ap_scores["n_total_pairs"].to_numpy(),
            pairs,
            null_size,
            seed,
            observed=map_scores["mean_average_precision"].to_numpy(),
            progress_bar=progress_bar,
        )
        return map_scores

    if null == "permutation":
        logger.info("Computing p-values...")
        map_scores["p_value"] = nulls.permutation.permutation_pvalues(
            map_scores["indices"].tolist(),
            profile,
            ap_scores["n_pos_pairs"].to_numpy(),
            ap_scores["n_total_pairs"].to_numpy(),
            pairs.pos,
            pairs.neg,
            pairs.similarity,
            null_size,
            seed,
            observed=map_scores["mean_average_precision"].to_numpy(),
            progress_bar=progress_bar,
        )
        return map_scores

    if method == "fast" and null == "copied":
        logger.info("Computing p-values...")
        map_scores["p_value"] = _copied_pvalues(
            map_scores["mean_average_precision"].to_numpy(),
            map_scores["indices"].tolist(),
            rev_ix,
            null_confs,
            null_size,
            seed,
            backend,
            progress_bar,
        )
        return map_scores

    if method == "fast":
        groups = map_scores["indices"].tolist()
        num_pos = ap_scores["n_pos_pairs"].to_numpy().astype(np.int64)
        num_neg = ap_scores["n_total_pairs"].to_numpy().astype(np.int64) - num_pos
        edges_of = _group_edges(groups, num_pos, profile, pos_pairs)
        signatures = [
            None
            if edges is None
            else nulls.joint.canonical_signature(zip(num_pos[ix], num_neg[ix]), edges)
            for ix, edges in zip(groups, edges_of)
        ]
        scores = map_scores["mean_average_precision"].to_numpy()
        p_values = np.empty(len(groups), dtype=np.float64)

        logger.info("Computing p-values...")
        seed = nulls.pvalues.resolve_seed(seed)
        unique = sorted(set(sig for sig in signatures if sig is not None))
        index = {sig: k for k, sig in enumerate(unique)}
        joint = [g for g, sig in enumerate(signatures) if sig is not None]
        p_values[joint] = nulls.joint.joint_pvalues(
            scores[joint],
            [index[signatures[g]] for g in joint],
            unique,
            null_size,
            seed,
            progress_bar=progress_bar,
        )

        legacy = [g for g, sig in enumerate(signatures) if sig is None]
        if legacy:
            warnings.warn(
                f"{len(legacy)} group(s) have positives that are not their other "
                "members; pass pos_pairs to sample their mAP null jointly. Falling "
                "back to the previous null, which copies one draw across members "
                "with equal pair counts.",
                stacklevel=2,
            )
            logger.info("Computing p-values...")
            p_values[legacy] = _copied_pvalues(
                scores[legacy],
                [groups[g] for g in legacy],
                rev_ix,
                null_confs,
                null_size,
                seed,
                backend,
                progress_bar,
            )
        map_scores["p_value"] = p_values
        return map_scores

    logger.info("Computing null_dist...")
    # Generate null distributions for each unique configuration
    null_dists = compute._get_null_dists_legacy(
        null_confs, null_size, seed, cache_dir, progress_bar
    )

    # Function to calculate the p-value for a mAP score based on the null distribution
    def get_p_value(params):
        map_score, indices = params
        null_dist = null_dists[rev_ix[indices]].mean(axis=0)
        num = (null_dist > map_score).sum()
        p_value = (num + 1) / (null_size + 1)  # Add 1 for stability
        return p_value

    logger.info("Computing p-values...")

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

    return map_scores


def _group_edges(groups, num_pos, profile, pos_pairs):
    """Positive pairs inside each group, as local ``(i, j)`` member pairs.

    Without ``pos_pairs``, a group whose members each have ``len(group) - 1``
    positives is taken to be a replicate group (all its pairs are positives),
    and other groups get None.
    """
    if pos_pairs is None:
        return [
            [(i, j) for i in range(len(ix)) for j in range(i + 1, len(ix))]
            if (num_pos[ix] == len(ix) - 1).all()
            else None
            for ix in groups
        ]
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    size = int(max(pos_pairs.max(initial=-1), profile.max(initial=-1))) + 1
    degree = np.bincount(pos_pairs.ravel(), minlength=size)
    wrong = np.flatnonzero(degree[profile] != num_pos)
    if len(wrong):
        m = profile[wrong[0]]
        raise ValueError(
            f"profile {m} has {degree[m]} positive pairs in pos_pairs but "
            f"{num_pos[wrong[0]]} in ap_scores; pos_pairs must index the rows of "
            "the frame average_precision returned, as "
            "average_precision(..., return_pairs=True) gives them"
        )
    group_of = np.full(size, -1)
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
    return edges_of


def _copied_pvalues(
    scores, groups, rev_ix, null_confs, null_size, seed, backend, progress_bar
):
    """P-values against the mean of the members' AP nulls (:func:`nulls.map_pvalues`).

    ``groups`` lists each group's rows; members sharing a configuration share
    null samples, as in copairs <= 0.5.5.
    """
    if not groups:
        return np.zeros(0, dtype=np.float64)
    gid = np.repeat(np.arange(len(groups)), [len(ix) for ix in groups])
    rows = np.concatenate(groups).astype(np.int64)
    keys, conf_cnt = np.unique(gid * len(null_confs) + rev_ix[rows], return_counts=True)
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
    null: Optional[str] = None,
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
        Number of workers used. Default defined by tqdm's `thread_map`.
    cache_dir : str or Path
        Location to save the cache (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"``, see :func:`get_map_pvalue`.
    backend : str
        Backend of the fast method, see :func:`get_map_pvalue`.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores; see `get_map_pvalue`.
        Needed when a group's positives are not its other members.
    null : str, optional
        Null of the group mAP, see :func:`get_map_pvalue`.
    pairs : RankPairs or LabelPairs, optional
        Pairs of the AP scores, needed by ``null="permutation"`` and attached
        to AP frames; see :func:`get_map_pvalue`.

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
    null: Optional[str] = None,
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
        Number of workers used. Default defined by tqdm's `thread_map`.
    cache_dir : str or Path
        Location to save the cache (``method="legacy"`` only).
    method : str
        ``"fast"`` (default) or ``"legacy"``, see :func:`get_map_pvalue`.
    backend : str
        Backend of the fast method, see :func:`get_map_pvalue`.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores; see `get_map_pvalue`.
        Needed when a group's positives are not its other members.
    null : str, optional
        Null of the group mAP, see :func:`get_map_pvalue`.
    pairs : RankPairs or LabelPairs, optional
        Pairs of the AP scores, needed by ``null="permutation"`` and attached
        to AP frames; see :func:`get_map_pvalue`.

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
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    chunksize = kwargs.pop("chunksize", 1)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, *iterables, chunksize=chunksize, **kwargs))
