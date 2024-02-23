import itertools
import logging

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import MatcherMultilabel, UnpairedException

from .filter import evaluate_and_filter, flatten_str_list, validate_pipeline_input

logger = logging.getLogger("copairs")


def create_neg_query_solver(neg_pairs, neg_sims):
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


def build_rank_lists_multi(pos_pairs, pos_sims, pos_counts, negs_for):
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
                np.ones(mpos_pairs.size, dtype=np.int32),
                np.zeros(len(neg_sims), dtype=np.int32),
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
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    multilabel_col,
    batch_size=20000,
) -> pd.DataFrame:
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)
    validate_pipeline_input(meta, feats, columns)
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info("Indexing metadata...")
    matcher = MatcherMultilabel(
        *evaluate_and_filter(meta, columns), multilabel_col=multilabel_col, seed=0
    )

    logger.info("Finding positive pairs...")
    pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_keys = pos_pairs.keys()
    pos_counts = np.fromiter(map(len, pos_pairs.values()), dtype=np.int32)
    pos_total = sum(pos_counts)
    if pos_total == 0:
        raise UnpairedException("Unable to find positive pairs.")
    pos_pairs = np.fromiter(
        itertools.chain.from_iterable(pos_pairs.values()),
        dtype=np.dtype((np.int32, 2)),
        count=pos_total,
    )

    logger.info("Finding negative pairs...")
    neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_total = sum(len(p) for p in neg_pairs.values())
    if neg_total == 0:
        raise UnpairedException("Unable to find any negative pairs.")
    neg_pairs = np.fromiter(
        itertools.chain.from_iterable(neg_pairs.values()),
        dtype=np.dtype((np.int32, 2)),
        count=neg_total,
    )

    logger.info("Dropping dups in negative pairs...")
    neg_pairs = np.unique(neg_pairs, axis=0)

    logger.info("Computing positive similarities...")
    pos_sims = compute.pairwise_cosine(feats, pos_pairs, batch_size)

    logger.info("Computing negative similarities...")
    neg_sims = compute.pairwise_cosine(feats, neg_pairs, batch_size)

    logger.info("Computing AP per label...")
    negs_for = create_neg_query_solver(neg_pairs, neg_sims)
    ap_scores_list, null_confs_list, ix_list = build_rank_lists_multi(
        pos_pairs, pos_sims, pos_counts, negs_for
    )

    logger.info("Creating result DataFrame...")
    results = []
    for i, key in enumerate(pos_keys):
        result = pd.DataFrame(
            {
                "average_precision": ap_scores_list[i],
                "n_pos_pairs": null_confs_list[i][:, 0],
                "n_total_pairs": null_confs_list[i][:, 1],
                "ix": ix_list[i],
            }
        )
        if isinstance(key, tuple):
            # Is a ComposedKey
            for k, v in zip(key._fields, key):
                result[k] = v
        else:
            result[multilabel_col] = key
        results.append(result)
    results = pd.concat(results).reset_index(drop=True)
    meta = meta.drop(multilabel_col, axis=1)
    results = meta.merge(results, right_on="ix", left_index=True).drop("ix", axis=1)
    results["n_pos_pairs"] = results["n_pos_pairs"].fillna(0).astype(np.int32)
    results["n_total_pairs"] = results["n_total_pairs"].fillna(0).astype(np.int32)
    logger.info("Finished.")
    return results
