import itertools
import logging

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import Matcher, UnpairedException

from .filter import evaluate_and_filter, flatten_str_list, validate_pipeline_input

logger = logging.getLogger("copairs")


def build_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims):
    labels = np.concatenate(
        [
            np.ones(pos_pairs.size, dtype=np.int32),
            np.zeros(neg_pairs.size, dtype=np.int32),
        ]
    )
    ix = np.concatenate([pos_pairs.ravel(), neg_pairs.ravel()])
    sim_all = np.concatenate([np.repeat(pos_sims, 2), np.repeat(neg_sims, 2)])
    ix_sort = np.lexsort([1 - sim_all, ix])
    rel_k_list = labels[ix_sort]
    paired_ix, counts = np.unique(ix, return_counts=True)
    return paired_ix, rel_k_list, counts


def average_precision(
    meta, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby, batch_size=20000
) -> pd.DataFrame:
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)
    validate_pipeline_input(meta, feats, columns)

    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()
    logger.info("Indexing metadata...")
    matcher = Matcher(*evaluate_and_filter(meta, columns), seed=0)

    logger.info("Finding positive pairs...")
    pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_total = sum(len(p) for p in pos_pairs.values())
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
        raise UnpairedException("Unable to find negative pairs.")
    neg_pairs = np.fromiter(
        itertools.chain.from_iterable(neg_pairs.values()),
        dtype=np.dtype((np.int32, 2)),
        count=neg_total,
    )

    logger.info("Computing positive similarities...")
    pos_sims = compute.pairwise_cosine(feats, pos_pairs, batch_size)

    logger.info("Computing negative similarities...")
    neg_sims = compute.pairwise_cosine(feats, neg_pairs, batch_size)

    logger.info("Building rank lists...")
    paired_ix, rel_k_list, counts = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims
    )

    logger.info("Computing average precision...")
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)

    logger.info("Creating result DataFrame...")
    meta["n_pos_pairs"] = 0
    meta["n_total_pairs"] = 0
    meta.loc[paired_ix, "average_precision"] = ap_scores
    meta.loc[paired_ix, "n_pos_pairs"] = null_confs[:, 0]
    meta.loc[paired_ix, "n_total_pairs"] = null_confs[:, 1]
    logger.info("Finished.")
    return meta


def p_values(dframe: pd.DataFrame, null_size: int, seed: int):
    """Compute p-values"""
    mask = dframe["n_pos_pairs"] > 0
    pvals = np.full(len(dframe), np.nan, dtype=np.float32)
    scores = dframe.loc[mask, "average_precision"].values
    null_confs = dframe.loc[mask, ["n_pos_pairs", "n_total_pairs"]].values
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed)
    return pvals
