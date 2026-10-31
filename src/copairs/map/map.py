"""Functions to compute mean average precision."""

import logging
from os import cpu_count
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from copairs import compute
from copairs.nulls import joint, permutation, tie_thresholds
from copairs.methods import check_null
from copairs.matching import UnpairedException
from copairs.nulls.common import (
    check_seed,
    check_pairs,
    check_null_size,
    check_max_workers,
)
from copairs.map.hierarchical_fdr import (
    check_group_columns,
    apply_fdr_correction,
    check_hierarchical_by,
    apply_hierarchical_fdr_correction,
)
from copairs.map.average_precision import (
    attached_pairs,
    check_recomputable,
)

logger = logging.getLogger("copairs")


def get_map_pvalue(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    seed: int | None = 0,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    pos_pairs: Optional[np.ndarray] = None,
    null: str = "permutation",
    pairs: dict | None = None,
    *,
    max_reference_profiles: int = 16_384,
    pool_memory_budget: int = 4 * 1024**3,
    key_table_memory_budget: int = 256 * 1024**2,
) -> pd.DataFrame:
    """Compute mAP scores and p-values from AP scores.

    This function groups AP scores by the specified columns, computes the mean
    Average Precision (mAP) for each group, and calculates p-values by comparing
    against the null distribution of the group's mAP chosen by `null`. A null
    mAP counts against the group when it is ``>=`` the group's mAP, so an mAP
    equal to an atom of the null (e.g. perfect retrieval) counts the atom; values
    within the scores' rounding count as ties (`copairs.nulls.tie_thresholds`).

    Parameters
    ----------
    ap_scores : pd.DataFrame
        DataFrame containing individual Average Precision (AP) scores and pair statistics
        (e.g., number of positive pairs `n_pos_pairs` and total pairs `n_total_pairs`).
    sameby : list or str
        Nonempty, distinct string column names present in the input.
    null_size : int
        Number of Monte Carlo draws. Zero requests scores only: p-values are NaN
        after input/layout validation, without sampling or null-only tables.
    seed : int or None, default 0
        Reproducible seed; None resolves fresh entropy once per run. Joint and
        permutation accept integers in [0, 2**64); copied accepts any integer >= 0.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    max_workers : int
        Threads that compute the copied null's p-values or run the
        permutation null's groups. Default defined by
        `concurrent.futures.ThreadPoolExecutor`.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples.
    pos_pairs : np.ndarray, optional
        The `(n_pairs, 2)` positive pairs used to compute the AP scores,
        indexing the rows of the frame `average_precision` returned (its
        `return_pairs=True` gives them, as `pairs["pos"]`); `ap_scores` must keep
        that frame's index. Pairs whose counts disagree with `n_pos_pairs`
        raise. When omitted, they are taken from ``pairs`` or, failing that,
        from the pairs `average_precision` attached to ``ap_scores``. Multi-member
        groups require actual positive pairs; counts alone do not identify their
        graph. Singletons can use their AP null without pairs. If a pair handle
        is available, explicit positive pairs must agree with it. Joint null only.
    null : str
        Null of the group mAP. ``"permutation"`` (default) relabels profiles
        (`copairs.nulls.permutation`): in each block of members ranked against
        the same references (e.g. a plate's replicates and controls), any of
        the block's profiles could have carried the group's label, and each
        draw recomputes the group's mAP from the observed similarities. All
        positive edges must stay inside the tested group; the graph need not
        be complete. Original metadata rules define the graph and reference
        pools and are not reapplied per draw. A singleton with external
        positives can use ``"joint"``, its AP null. Positive draw counts use
        Monte Carlo sampling with (hits + 1)/(draws + 1). It recomputes similarities, so
        APs computed with a distance that estimates its scale per batch
        (``"seuclidean"``, ``"mahalanobis"``) raise. Multilabel pair dictionaries
        use negative-graph components, preserving member counts per component
        and other labels, with the original positive graph. Negatives follow
        the original negative/lone-pair structure; counts can change. Each
        group must contain one complete label. Overlapping reference sets are
        allowed for this explicitly distinct multilabel model. ``"joint"`` samples
        the members' APs together (`copairs.nulls.joint`): a positive pair
        inside the group is one random number in both members' rank lists,
        and other similarities are independent, which real profiles are not
        (members share references, and some profiles are close to many
        others), so it can give too small p-values for groups of many
        replicates. ``"copied"`` samples each member's AP null per
        `(n_pos_pairs, n_total_pairs)`
        configuration and the group's null averages them, so members with
        equal pair counts share one draw and the null does not narrow as
        replicates are added.
    pairs : dict, optional
        Dictionary with ``"pos"``, ``"neg"`` and ``"similarity"`` entries, as
        ``average_precision(..., return_pairs=True)`` returns them; their
        indices are rows of that call's frame, so ``ap_scores`` must keep its
        index. The permutation null needs them and, when not given, takes
        them from ``ap_scores.attrs["copairs_pairs"]``, where
        `average_precision` attaches them; frames that lost them (built,
        reloaded, pickled, or concatenated from several AP calls) need
        ``pairs=`` or another null. The joint null takes its positive pairs
        from ``pairs`` too, or from the attached pairs (see ``pos_pairs``).
        Multilabel dictionaries also carry ``"label_col"``, ``"profile"``,
        ``"row_key"``, ``"keys"`` and the lazy ``"_lone"`` pair function;
        ``"label_col"`` selects the multilabel permutation path.
        Explicit pairs override attachments. Keep the original index, matching
        pairs and complete tested groups/labels; validation checks structure,
        not source identity or group completeness. Recompute AP when removing
        group members. Copies share the dictionary, features and arrays; keep
        them unchanged between AP and mAP. Custom similarities must return the
        same value for a pair regardless of batch size or contents.
    max_reference_profiles : int, default 16384
        Maximum total reference profiles across a tested group's blocks; >= 0.
        For multilabel groups, component-union size minus its members; all
        components count. Pool-table estimates include members too.
        Checked before quadratic pool tables are allocated.
    pool_memory_budget : int, default 4294967296
        Positive byte budget for estimated concurrent pool key/rank tables.
        One oversized pool may run alone. This does not cap total process RAM.
    key_table_memory_budget : int, default 268435456
        Byte limit for optional cross-pool key reuse; zero disables it.
        Neither budget includes features, layouts or all temporary arrays.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - Columns from `sameby` (group identifiers).
        - `mean_average_precision`: Mean AP score for each group.
        - `mean_normalized_average_precision`: Mean normalized AP score (scale-independent).
        - `p_value`: p-value comparing mAP to the null distribution.
        - `indices`: The group's rows, as positions in ap_scores after rows
          with a NaN AP or no positive pairs are dropped.

    Raises
    ------
    UnpairedException
        If no row has a non-NaN AP, positive pairs and a value in every
        `sameby` column, e.g. an empty table.
    ValueError
        Invalid grouping, pair dictionaries/counts, layout or resource arguments.
        ``max_workers`` must be a positive integer or None even for zero draws
        and joint sampling, which does not use that executor.

    """
    null = check_null(null)
    null_size = check_null_size(null_size)
    sameby = check_group_columns(sameby, "sameby", ap_scores)
    max_workers = check_max_workers(max_workers)
    resources = dict(
        zip(
            ("max_reference_profiles", "pool_memory_budget", "key_table_memory_budget"),
            permutation.check_resources(
                max_reference_profiles, pool_memory_budget, key_table_memory_budget
            ),
        )
    )
    if null != "copied":
        check_seed(seed)
    elif seed is not None and (int(seed) != seed or seed < 0):  # NumPy: any int >= 0
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if null != "joint" and pos_pairs is not None:
        raise ValueError("pos_pairs is used by the joint null only")
    if null != "copied" and pairs is None:
        pairs = attached_pairs(ap_scores)
    if pairs is not None and null != "copied":
        if (
            not isinstance(pairs, dict)
            or not {"pos", "neg", "similarity"} <= pairs.keys()
        ):
            raise ValueError(
                "pairs must be a dictionary with pos, neg and similarity entries"
            )
        if pos_pairs is not None and not np.array_equal(
            check_pairs(pos_pairs, "pos_pairs"),
            check_pairs(pairs["pos"], "positive pairs"),
        ):
            raise ValueError(
                "pos_pairs must index the rows of the AP frame and match the selected handle's positive pairs"
            )
    if null == "joint" and pos_pairs is None and pairs is not None:
        pos_pairs = pairs["pos"]
    if null == "permutation" and pairs is None:
        raise ValueError(
            "null='permutation' (the default) needs the pairs the APs were "
            "computed from. average_precision attaches them to the frame it "
            "returns, and these AP scores have none: they were built, loaded "
            "from disk, pickled or concatenated from several AP calls. "
            "Both average_precision and multilabel.average_precision attach pairs. "
            "Recompute the APs, pass pairs=, or choose null='joint' "
            "or null='copied'."
        )
    if null == "permutation":
        check_recomputable(pairs["similarity"])

    if null == "permutation" or pos_pairs is not None:
        _check_row_index(ap_scores.index)

    # Filter out invalid or incomplete AP scores, remembering the original rows.
    # Rows missing a sameby value stay (the groupby drops them), so `indices`
    # keep their positions; the check counts only rows a group can take.
    valid = (
        ap_scores["average_precision"].notna() & (ap_scores["n_pos_pairs"] > 0)
    ).fillna(False)
    if not (valid & ap_scores[sameby].notna().all(axis=1)).any():
        raise UnpairedException(
            "Unable to find valid AP scores: no row has a non-NaN AP, positive "
            "pairs and a value in every sameby column."
        )
    missing_keys = valid & ~ap_scores[sameby].notna().all(axis=1)
    tested = ap_scores.loc[valid & ~missing_keys]
    counts = _normalize_pair_counts(tested)
    ap_values = tested["average_precision"].to_numpy(dtype=np.float64)
    if not np.isfinite(ap_values).all() or ((ap_values < 0) | (ap_values > 1)).any():
        raise ValueError("usable AP scores must be finite values in [0, 1]")
    if (~valid).any() or missing_keys.any():
        logger.warning(
            "Excluding %d rows with missing AP/no positive pairs and %d rows with missing group keys",
            int((~valid).sum()),
            int(missing_keys.sum()),
        )
    ap_scores = ap_scores[valid]
    profile = ap_scores.index.to_numpy()
    ap_scores = ap_scores.reset_index(drop=True)
    # Keep positional indices for rows without group keys, but never cast their
    # unused counts (which may be missing/invalid) or pass them into a sampler.
    normalized = np.zeros((len(ap_scores), 2), dtype=np.int64)
    normalized[(~missing_keys[valid]).to_numpy(dtype=bool)] = counts
    ap_scores[["n_pos_pairs", "n_total_pairs"]] = normalized

    # Average the APs in float64 and round the means to the APs' dtype: ties
    # carry the APs' rounding, and pandas averages float32 APs to float32 or
    # float64 depending on its version and the group keys. The reported mAP
    # is then the score the nulls count ties against, under any pandas.
    ap_dtype = getattr(
        ap_scores["average_precision"].dtype,
        "numpy_dtype",
        ap_scores["average_precision"].to_numpy().dtype,
    )
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
    )

    if null == "copied":
        map_scores["p_value"] = (
            np.full(len(groups), np.nan)
            if null_size == 0
            else _copied_pvalues(scores, groups, ap_scores, **copied)
        )
        return map_scores

    if null == "permutation":
        logger.info("Computing p-values...")
        arguments = (
            groups,
            profile,
            ap_scores["n_pos_pairs"].to_numpy(),
            ap_scores["n_total_pairs"].to_numpy(),
        )
        options = dict(
            observed=scores,
            progress_bar=progress_bar,
            max_workers=max_workers,
            **resources,
        )
        if "label_col" in pairs:
            pvalues = permutation.label_permutation_pvalues(
                *arguments, pairs, null_size, seed, **options
            )
        else:
            pvalues = permutation.permutation_pvalues(
                *arguments,
                pairs["pos"],
                pairs["neg"],
                pairs["similarity"],
                null_size,
                seed,
                **options,
            )
        map_scores["p_value"] = pvalues
        return map_scores

    num_pos = ap_scores["n_pos_pairs"].to_numpy().astype(np.int64)
    num_neg = ap_scores["n_total_pairs"].to_numpy().astype(np.int64) - num_pos
    logger.info("Computing p-values...")
    map_scores["p_value"] = joint.group_pvalues(
        scores, groups, num_pos, num_neg, null_size, seed, profile, pos_pairs,
        progress_bar=progress_bar,
    )  # fmt: skip
    return map_scores


def _normalize_pair_counts(frame):
    """Validate numeric counts without float-rounding integers; return int64 arrays."""
    error = "AP pair counts must be finite non-negative int64 integers with n_total_pairs >= n_pos_pairs"
    result = np.empty((len(frame), 2), dtype=np.int64)
    for i, name in enumerate(("n_pos_pairs", "n_total_pairs")):
        column = frame[name]
        if column.isna().any() or pd.api.types.is_bool_dtype(column.dtype):
            raise ValueError(error)
        if pd.api.types.is_integer_dtype(column.dtype):
            if len(column) and (int(column.min()) < 0 or int(column.max()) >= 2**63):
                raise ValueError(error)
        elif pd.api.types.is_float_dtype(column.dtype):
            values = column.to_numpy(dtype=np.float64)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values >= 2**63).any()
                or (values != np.floor(values)).any()
            ):
                raise ValueError(error)
        else:
            raise ValueError(error)
        result[:, i] = column.to_numpy(dtype=np.int64)
    if (result[:, 0] > result[:, 1]).any():
        raise ValueError(error)
    return result


def _check_row_index(index: pd.Index) -> None:
    """Raise unless ``index`` holds distinct row numbers, which AP pairs can index."""
    if index.dtype.kind not in "iu" or (len(index) and index.min() < 0):
        raise ValueError(
            "the AP scores' index must hold the row numbers of the frame "
            "average_precision returned, which the pairs index; got an index "
            f"of dtype {index.dtype}"
            + ("" if index.dtype.kind not in "iu" else " with negative labels")
            + ". Keep that frame's index (no set_index or reset_index)."
        )
    if not index.is_unique:
        raise ValueError(
            "the AP scores' index repeats row numbers, e.g. after concatenating "
            "copies or overlapping subsets of one AP frame, which keeps its "
            "pairs; the pairs-based nulls need each profile's AP once."
        )


def _copied_pvalues(
    scores, groups, ap_scores, null_size, seed, cache_dir, progress_bar, max_workers
):
    """P-values against the mean of the members' AP nulls, as in copairs <= 0.5.5.

    ``groups`` lists each group's rows of ``ap_scores``. Each configuration's
    AP null comes from :func:`compute.get_null_dists`, so members with equal
    pair counts share samples. Null mAPs count when ``>=`` the score, within
    float32 rounding: the AP samples are float32, so a null mAP equal to the
    score can lie up to that rounding below it.
    """
    rows = np.unique(np.concatenate(groups))
    null_confs = ap_scores.loc[rows, ["n_pos_pairs", "n_total_pairs"]].values
    null_confs, conf_ix = np.unique(null_confs, axis=0, return_inverse=True)
    rev_ix = np.full(len(ap_scores), -1, dtype=np.int64)
    rev_ix[rows] = conf_ix.ravel()

    logger.info("Computing null_dist...")
    null_dists = compute.get_null_dists(
        null_confs, null_size, seed=seed, cache_dir=cache_dir, progress_bar=progress_bar
    )

    thresholds = tie_thresholds(np.asarray(scores, dtype=np.float32))

    def get_p_value(k):
        null_dist = null_dists[rev_ix[groups[k]]].mean(axis=0, dtype=np.float64)
        num = (null_dist >= thresholds[k]).sum()
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
    seed: int | None = 0,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    pos_pairs: Optional[np.ndarray] = None,
    null: str = "permutation",
    pairs: dict | None = None,
    *,
    max_reference_profiles: int = 16_384,
    pool_memory_budget: int = 4 * 1024**3,
    key_table_memory_budget: int = 256 * 1024**2,
) -> pd.DataFrame:
    """Calculate the Mean Average Precision (mAP) score and associated p-values.

    This function computes the Mean Average Precision (mAP) score by grouping profiles
    based on the specified criteria (`sameby`). It calculates the significance of mAP
    scores by comparing them to a null distribution and performs multiple testing
    corrections using Benjamini-Hochberg FDR.

    Per-call resource controls (``max_reference_profiles``, ``pool_memory_budget``,
    ``key_table_memory_budget``), pair dictionaries and supported layouts follow
    :func:`get_map_pvalue`. With ``null_size=0`` statistical columns are NaN and
    significance flags are nullable booleans containing pd.NA; correction is skipped.

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
    seed : int or None, default 0
        Reproducible seed; None requests fresh randomness once per run.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    max_workers : int
        Threads that compute the copied null's p-values or run the
        permutation null's groups. Default defined by
        `concurrent.futures.ThreadPoolExecutor`.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores, for ``null="joint"``;
        see `get_map_pvalue`. Needed there when a group's positives are not
        its other members.
    null : str
        Null of the group mAP: ``"permutation"`` (default), ``"joint"`` or
        ``"copied"``; see `get_map_pvalue`.
    pairs : dict, optional
        Pair dictionary of the AP scores, needed by ``null="permutation"`` and
        attached to AP frames; see `get_map_pvalue`.

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

    Raises
    ------
    UnpairedException
        If no row has a non-NaN AP, positive pairs and a value in every
        `sameby` column, e.g. an empty table.

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
        null=null,
        pairs=pairs,
        max_reference_profiles=max_reference_profiles,
        pool_memory_budget=pool_memory_budget,
        key_table_memory_budget=key_table_memory_budget,
    )

    if null_size == 0:
        map_scores["corrected_p_value"] = np.nan
    else:
        map_scores = apply_fdr_correction(map_scores)
    return _mark_significance(map_scores, threshold)


def mean_average_precision_hierarchical(
    ap_scores: pd.DataFrame,
    sameby: List[str],
    null_size: int,
    threshold: float,
    seed: int | None = 0,
    hierarchical_by: Optional[List[str]] = None,
    progress_bar: bool = True,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    pos_pairs: Optional[np.ndarray] = None,
    null: str = "permutation",
    pairs: dict | None = None,
    *,
    max_reference_profiles: int = 16_384,
    pool_memory_budget: int = 4 * 1024**3,
    key_table_memory_budget: int = 256 * 1024**2,
) -> pd.DataFrame:
    """Calculate the Mean Average Precision (mAP) score with hierarchical FDR correction.

    This function computes the Mean Average Precision (mAP) score by grouping profiles
    based on the specified criteria (`sameby`). It applies hierarchical FDR correction
    appropriate for grouped hypothesis testing, such as dose-response data.

    Per-call resource controls (``max_reference_profiles``, ``pool_memory_budget``,
    ``key_table_memory_budget``), pair dictionaries and supported layouts follow
    :func:`get_map_pvalue`. With ``null_size=0`` all statistical columns, including
    stage-1 p-values, are NaN; significance flags contain pd.NA and correction is skipped.
    Supply ``hierarchical_by`` explicitly, preferably as a keyword.

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
    seed : int or None, default 0
        Reproducible seed; None requests fresh randomness once per run.
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
        Threads that compute the copied null's p-values or run the
        permutation null's groups. Default defined by
        `concurrent.futures.ThreadPoolExecutor`.
    cache_dir : str or Path
        Location to save the cache of the copied null's AP samples.
    pos_pairs : np.ndarray, optional
        Positive pairs used to compute the AP scores, for ``null="joint"``;
        see `get_map_pvalue`. Needed there when a group's positives are not
        its other members.
    null : str
        Null of the group mAP: ``"permutation"`` (default), ``"joint"`` or
        ``"copied"``; see `get_map_pvalue`.
    pairs : dict, optional
        Pair dictionary of the AP scores, needed by ``null="permutation"`` and
        attached to AP frames; see `get_map_pvalue`.

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

    Raises
    ------
    UnpairedException
        If no row has a non-NaN AP, positive pairs and a value in every
        `sameby` column, e.g. an empty table.

    See Also
    --------
    mean_average_precision : For standard BH FDR correction.

    """
    if hierarchical_by is None:
        raise ValueError("hierarchical_by is required for hierarchical correction")
    sameby = check_group_columns(sameby, "sameby", ap_scores)
    hierarchical_by = check_group_columns(hierarchical_by, "hierarchical_by")
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
        pos_pairs=pos_pairs,
        null=null,
        pairs=pairs,
        max_reference_profiles=max_reference_profiles,
        pool_memory_budget=pool_memory_budget,
        key_table_memory_budget=key_table_memory_budget,
    )

    if null_size == 0:
        for column in (
            "corrected_p_value",
            "stage1_p_value",
            "stage1_corrected_p_value",
        ):
            map_scores[column] = np.nan
        map_scores["stage1_significant"] = pd.array(
            [pd.NA] * len(map_scores), dtype="boolean"
        )
    else:
        map_scores = apply_hierarchical_fdr_correction(
            map_scores, hierarchical_by, sameby
        )
    return _mark_significance(map_scores, threshold)


def _mark_significance(map_scores, threshold):
    """Keep significance unknown for untested rows instead of comparing NaN."""
    for score, flag in (
        ("p_value", "below_p"),
        ("corrected_p_value", "below_corrected_p"),
    ):
        significant = map_scores[score] < threshold
        if map_scores[score].isna().any():
            significant = significant.astype("boolean").mask(map_scores[score].isna())
        map_scores[flag] = significant
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
