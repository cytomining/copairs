"""Functions to compute mAP with multilabel support."""

import logging
from typing import List

import numpy as np
import pandas as pd

from copairs import fastap, compute
from copairs.fastap import multilabel as fast_multilabel
from copairs.matching import (
    UnpairedException,
    sorted_unique_pairs,
    find_pairs_multilabel,
)
from copairs.fastap.ranking import _host

from .filter import flatten_str_list, evaluate_and_filter, validate_pipeline_input
from .normalization import normalize_ap
from .average_precision import LabelPairs

logger = logging.getLogger("copairs")


def _create_neg_query_solver(neg_pairs, neg_sims):
    # Melting and sorting by ix. neg_cutoffs splits the contiguous array
    neg_ix = neg_pairs.ravel()
    neg_sims = np.repeat(neg_sims, 2)

    sort_ix = np.argsort(neg_ix)
    neg_sims = neg_sims[sort_ix]

    neg_ix, neg_counts = np.unique(neg_ix, return_counts=True)
    neg_cutoffs = compute.to_cutoffs(neg_counts)

    def negs_for(query: np.ndarray):
        locs = np.searchsorted(neg_ix, query)
        sizes = neg_counts[locs]
        start = neg_cutoffs[locs]
        end = start + sizes
        slices = compute.concat_ranges(start, end)
        batch_sims = neg_sims[slices]
        return batch_sims, sizes

    return negs_for


def _build_rank_lists_multi(pos_pairs, pos_sims, pos_counts, negs_for):
    ap_scores_list, null_confs_list, ix_list = [], [], []

    start = 0
    for end in pos_counts.cumsum():
        mpos_pairs = pos_pairs[start:end]
        mpos_sims = pos_sims[start:end]
        start = end
        query = np.unique(mpos_pairs)
        neg_sims, neg_counts = negs_for(query)
        neg_ix = np.repeat(query, neg_counts)
        labels = np.concatenate(
            [
                np.ones(mpos_pairs.size, dtype=np.uint32),
                np.zeros(len(neg_sims), dtype=np.uint32),
            ]
        )

        ix = np.concatenate([mpos_pairs.ravel(), neg_ix])
        sim_all = np.concatenate([np.repeat(mpos_sims, 2), neg_sims])
        ix_sort = np.lexsort([1 - sim_all, ix])
        rel_k_list = labels[ix_sort]
        _, counts = np.unique(ix, return_counts=True)
        ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
        ap_scores_list.append(ap_scores)
        null_confs_list.append(null_confs)
        ix_list.append(query)
    return ap_scores_list, null_confs_list, ix_list


def average_precision(
    meta: pd.DataFrame,
    feats: pd.DataFrame,
    pos_sameby: List[str],
    pos_diffby: List[str],
    neg_sameby: List[str],
    neg_diffby: List[str],
    multilabel_col,
    batch_size=20000,
    distance="cosine",
    progress_bar: bool = True,
    method: str = "fast",
    backend: str = "auto",
    return_pairs: bool = False,
) -> pd.DataFrame:
    """
    Compute average precision with multilabel support.

    Returns normalized_average_precision in addition to average_precision.
    ``method`` and ``backend`` are as in :func:`copairs.map.average_precision`;
    the fast method computes per-label APs on the CPU with Numba (``"cuda"``
    accelerates similarities only). With ``return_pairs``, also returns the
    :class:`~copairs.map.LabelPairs` the APs were computed from, which
    ``null="permutation"`` in :func:`copairs.map.mean_average_precision`
    needs.

    See Also
    --------
    copairs.map.average_precision : Average precision without multilabel support.
    """
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)
    backend, similarity = fastap.setup(
        method, backend, feats, distance, batch_size, progress_bar
    )
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info("Indexing metadata...")

    logger.info("Finding positive pairs...")
    pos_pairs, keys, pos_counts = find_pairs_multilabel(
        meta,
        sameby=pos_sameby,
        diffby=pos_diffby,
        multilabel_col=multilabel_col,
        method=method,
    )
    if len(pos_pairs) == 0:
        raise UnpairedException("Unable to find positive pairs.")

    logger.info("Finding negative pairs...")
    neg_pairs = find_pairs_multilabel(
        meta,
        sameby=neg_sameby,
        diffby=neg_diffby,
        multilabel_col=multilabel_col,
        method=method,
    )
    if len(neg_pairs) == 0:
        raise UnpairedException("Unable to find any negative pairs.")
    if method == "legacy":  # the fast matchers return sorted unique pairs
        logger.info("Dropping dups in negative pairs...")
        neg_pairs = sorted_unique_pairs(neg_pairs)

    logger.info("Computing positive similarities...")
    pos_sims = similarity(pos_pairs)

    logger.info("Computing negative similarities...")
    neg_sims = similarity(neg_pairs)

    def pairs_of(profile):
        lone = lone_key = None
        if multilabel_col in neg_diffby and multilabel_col not in neg_sameby:
            lone, lone_key = _lone_pairs(
                meta, multilabel_col, neg_sameby, neg_diffby, method, keys
            )
        return LabelPairs(
            np.asarray(pos_pairs),
            np.asarray(neg_pairs),
            lambda p: _host(similarity(p)),
            profile=np.asarray(profile, dtype=np.int64),
            label_col=multilabel_col,
            keys=np.asarray(keys),
            pos_key=np.repeat(np.arange(len(pos_counts)), pos_counts),
            lone=lone,
            lone_key=lone_key,
        )

    if method == "fast":
        logger.info("Computing AP per label...")
        label, ix, ap, num_pos, total = fast_multilabel.multilabel_ap(
            pos_pairs, pos_sims, pos_counts, neg_pairs, neg_sims, len(meta)
        )
        # The legacy per-label frames broadcast each key: NumPy keys keep their
        # dtype, strings make a str column (from a list, not an object array).
        label_values = np.asarray(keys)[label]
        if label_values.dtype == object:
            label_values = label_values.tolist()
        results = pd.DataFrame(
            {
                "average_precision": ap,
                "normalized_average_precision": normalize_ap(
                    ap, num_pos, total - num_pos
                ),
                "n_pos_pairs": num_pos,
                "n_total_pairs": total,
                "ix": ix,
                multilabel_col: label_values,
            }
        )
        results = _merge_results(meta, results, multilabel_col)
        return (results, pairs_of(ix)) if return_pairs else results

    logger.info("Computing AP per label...")
    negs_for = _create_neg_query_solver(neg_pairs, neg_sims)
    ap_scores_list, null_confs_list, ix_list = _build_rank_lists_multi(
        pos_pairs, pos_sims, pos_counts, negs_for
    )

    logger.info("Creating result DataFrame...")
    results = []
    "Here the positive pairs are per-item inside multilabel_col"
    # TODO Check if multi-label key is necessary
    for i, key in enumerate(keys):
        # Compute normalized AP for this label group
        M = null_confs_list[i][:, 0]  # n_pos_pairs
        L = null_confs_list[i][:, 1]  # n_total_pairs
        N = L - M  # n_neg_pairs
        normalized_scores = normalize_ap(ap_scores_list[i], M, N)

        result = pd.DataFrame(
            {
                "average_precision": ap_scores_list[i],
                "normalized_average_precision": normalized_scores,
                "n_pos_pairs": null_confs_list[i][:, 0],
                "n_total_pairs": null_confs_list[i][:, 1],
                "ix": ix_list[i],
                multilabel_col: key,
            }
        )
        results.append(result)
    results = pd.concat(results).reset_index(drop=True)
    merged = _merge_results(meta, results, multilabel_col)
    return (merged, pairs_of(np.concatenate(ix_list))) if return_pairs else merged


def _lone_pairs(meta, multilabel_col, neg_sameby, neg_diffby, method, keys):
    """Pairs sharing exactly one label that the negative rules join without it.

    Returns ``(pairs, key)``, ``key`` indexing ``keys``; labels without
    positive pairs are dropped.
    """
    other = [c for c in neg_diffby if c != multilabel_col]
    found = find_pairs_multilabel(
        meta,
        sameby=[multilabel_col, *neg_sameby],
        diffby=other,
        multilabel_col=multilabel_col,
        method=method,
    )
    pairs, found_keys, counts = found
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return pairs, np.zeros(0, dtype=np.int64)
    index = {k: i for i, k in enumerate(np.asarray(keys).tolist())}
    key = np.repeat(
        [index.get(k, -1) for k in np.asarray(found_keys).tolist()], counts
    ).astype(np.int64)
    a, b = pairs.min(axis=1), pairs.max(axis=1)
    _, inverse, shared = np.unique(
        a * len(meta) + b, return_inverse=True, return_counts=True
    )
    keep = (shared[inverse.ravel()] == 1) & (key >= 0)
    return np.stack([a[keep], b[keep]], axis=1), key[keep]


def _merge_results(meta, results, multilabel_col):
    meta = meta.drop(multilabel_col, axis=1)
    results = meta.merge(results, right_on="ix", left_index=True).drop("ix", axis=1)
    results["n_pos_pairs"] = results["n_pos_pairs"].fillna(0).astype(np.uint32)
    results["n_total_pairs"] = results["n_total_pairs"].fillna(0).astype(np.uint32)
    logger.info("Finished.")
    return results
