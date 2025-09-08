"""Functions to compute mAP with multilabel support."""

import logging
from typing import List

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import UnpairedException, find_pairs_multilabel

from .filter import flatten_str_list, evaluate_and_filter, validate_pipeline_input
from .normalization import normalize_ap

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
) -> pd.DataFrame:
    """
    Compute average precision with multilabel support.

    Returns normalized_average_precision in addition to average_precision.

    See Also
    --------
    copairs.map.average_precision : Average precision without multilabel support.
    """
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)
    distance_fn = compute.get_similarity_fn(distance, progress_bar=progress_bar)
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info("Indexing metadata...")

    logger.info("Finding positive pairs...")
    pos_pairs, keys, pos_counts = find_pairs_multilabel(
        meta, sameby=pos_sameby, diffby=pos_diffby, multilabel_col=multilabel_col
    )
    if len(pos_pairs) == 0:
        raise UnpairedException("Unable to find positive pairs.")

    logger.info("Finding negative pairs...")
    neg_pairs = find_pairs_multilabel(
        meta, sameby=neg_sameby, diffby=neg_diffby, multilabel_col=multilabel_col
    )
    if len(neg_pairs) == 0:
        raise UnpairedException("Unable to find any negative pairs.")

    logger.info("Dropping dups in negative pairs...")
    neg_pairs = np.unique(neg_pairs, axis=0)

    logger.info("Computing positive similarities...")
    pos_sims = distance_fn(feats, pos_pairs, batch_size)

    logger.info("Computing negative similarities...")
    neg_sims = distance_fn(feats, neg_pairs, batch_size)

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
    meta = meta.drop(multilabel_col, axis=1)
    results = meta.merge(results, right_on="ix", left_index=True).drop("ix", axis=1)
    results["n_pos_pairs"] = results["n_pos_pairs"].fillna(0).astype(np.uint32)
    results["n_total_pairs"] = results["n_total_pairs"].fillna(0).astype(np.uint32)
    logger.info("Finished.")
    return results
