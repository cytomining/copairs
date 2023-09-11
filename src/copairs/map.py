import itertools
import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs.compute import cosine_indexed
from copairs import compute_np
from copairs.matching import Matcher, MatcherMultilabel

logger = logging.getLogger('copairs')


def flatten_str_list(*args):
    '''create a single list with all the params given'''
    columns = set()
    for col in args:
        if isinstance(col, str):
            columns.add(col)
        else:
            columns.update(col)
    columns = list(columns)
    return columns


def create_matcher(obs: pd.DataFrame,
                   pos_sameby,
                   pos_diffby,
                   neg_sameby,
                   neg_diffby,
                   multilabel_col=None):
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)
    if multilabel_col:
        return MatcherMultilabel(obs, columns, multilabel_col, seed=0)
    return Matcher(obs, columns, seed=0)


def aggregate(result: pd.DataFrame, sameby, threshold: float) -> pd.DataFrame:
    agg_rs = result.groupby(sameby, as_index=False).agg({
        'average_precision':
        'mean',
        'p_value':
        lambda p_values: -np.log10(p_values).mean(),
    })
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        10**-agg_rs['p_value'], method='fdr_bh')
    agg_rs['q_value'] = pvals_corrected
    agg_rs['nlog10qvalue'] = (-np.log10(agg_rs['q_value']))
    agg_rs.rename({'p_value': 'nlog10pvalue'}, axis=1, inplace=True)
    agg_rs['above_p_threshold'] = agg_rs['nlog10pvalue'] > -np.log10(threshold)
    agg_rs['above_q_threshold'] = agg_rs['nlog10qvalue'] > -np.log10(threshold)
    agg_rs.rename(columns={'average_precision': 'mean_average_precision'}, inplace=True)
    return agg_rs


def build_rank_lists(pos_pairs, neg_pairs, pos_dists, neg_dists):
    labels = np.concatenate([np.ones(pos_pairs.size, dtype=np.int32),
                             np.zeros(neg_pairs.size, dtype=np.int32)])
    ix = np.concatenate([pos_pairs.ravel(), neg_pairs.ravel()])
    dist_all = np.concatenate([np.repeat(pos_dists, 2), np.repeat(neg_dists, 2)])
    ix_sort = np.lexsort([1 - dist_all, ix])
    rel_k_list = labels[ix_sort]
    _, counts = np.unique(ix, return_counts=True)
    return rel_k_list, counts


def run_pipeline(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    null_size,
    batch_size=20000,
    seed=0
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()
    logger.info('Indexing metadata...')
    matcher = create_matcher(meta, pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    logger.info('Finding positive pairs...')
    pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_total = sum(len(p) for p in pos_pairs.values())
    pos_pairs = np.fromiter(itertools.chain.from_iterable(pos_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=pos_total)

    logger.info('Finding negative pairs...')
    neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    total_neg = sum(len(p) for p in neg_pairs.values())
    neg_pairs = np.fromiter(itertools.chain.from_iterable(neg_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=total_neg)

    logger.info('Computing positive similarities...')
    pos_dists = cosine_indexed(feats, pos_pairs, batch_size).astype(np.float32)

    logger.info('Computing negative similarities...')
    neg_dists = cosine_indexed(feats, neg_pairs, batch_size).astype(np.float32)

    logger.info('Building rank lists...')
    rel_k_list, counts = build_rank_lists(pos_pairs, neg_pairs, pos_dists, neg_dists)

    logger.info('Computing average precision...')
    ap_scores, null_confs = compute_np.compute_ap_contiguos(rel_k_list, counts)

    logger.info('Computing p-values...')
    p_values = compute_np.compute_p_values(ap_scores, null_confs, null_size, seed=seed)

    logger.info('Creating result DataFrame...')
    meta['average_precision'] = ap_scores
    meta['p_value'] = p_values
    logger.info('Finished.')
    return meta


def create_neg_query_solver(neg_pairs, neg_dists):
    # Melting and sorting by ix. neg_cutoffs splits the contiguous array
    neg_ix = neg_pairs.ravel()
    neg_dists = np.repeat(neg_dists, 2)

    sort_ix = np.argsort(neg_ix)
    neg_dists = neg_dists[sort_ix]

    neg_ix, neg_counts = np.unique(neg_ix, return_counts=True)
    neg_cutoffs = compute_np.to_cutoffs(neg_counts)

    def negs_for(query: np.ndarray):
        locs = np.searchsorted(neg_ix, query)
        sizes = neg_counts[locs]
        start = neg_cutoffs[locs]
        end = start + sizes
        slices = compute_np.concat_ranges(start, end)
        batch_dists = neg_dists[slices]
        return batch_dists, sizes
    return negs_for


def build_rank_lists_multi(pos_pairs, pos_dists, pos_counts, negs_for, null_size, seed):
    ap_scores_list, p_values_list, ix_list = [], [], []

    start = 0
    for end in pos_counts.cumsum():
        mpos_pairs = pos_pairs[start:end]
        mpos_dists = pos_dists[start:end]
        start = end
        query = np.unique(mpos_pairs)
        neg_dists, neg_counts = negs_for(query)
        neg_ix = np.repeat(query, neg_counts)
        labels = np.concatenate([np.ones(mpos_pairs.size, dtype=np.int32),
                                 np.zeros(len(neg_dists), dtype=np.int32)])

        ix = np.concatenate([mpos_pairs.ravel(), neg_ix])
        dist_all = np.concatenate([np.repeat(mpos_dists, 2), neg_dists])
        ix_sort = np.lexsort([1 - dist_all, ix])
        rel_k_list = labels[ix_sort]
        _, counts = np.unique(ix, return_counts=True)
        ap_scores, null_confs = compute_np.compute_ap_contiguos(rel_k_list, counts)
        p_values = compute_np.compute_p_values(ap_scores, null_confs, null_size, seed=seed)

        ap_scores_list.append(ap_scores)
        p_values_list.append(p_values)
        ix_list.append(query)
    return ap_scores_list, p_values_list, ix_list


def run_pipeline_multilabel(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    null_size,
    multilabel_col,
    batch_size=20000,
    seed=0
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info('Indexing metadata...')
    matcher = create_matcher(meta, pos_sameby, pos_diffby, neg_sameby,
                             neg_diffby, multilabel_col)
    logger.info('Finding positive pairs...')
    pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_keys = pos_pairs.keys()
    pos_counts = np.fromiter(map(len, pos_pairs.values()), dtype=np.int32)
    pos_total = sum(pos_counts)
    pos_pairs = np.fromiter(itertools.chain.from_iterable(pos_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=pos_total)

    logger.info('Finding negative pairs...')
    neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    total_neg = sum(len(p) for p in neg_pairs.values())
    neg_pairs = np.fromiter(itertools.chain.from_iterable(neg_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=total_neg)

    logger.info('Dropping dups in negative pairs...')
    neg_pairs = np.unique(neg_pairs, axis=0)

    logger.info('Computing positive similarities...')
    pos_dists = cosine_indexed(feats, pos_pairs, batch_size).astype(np.float32)

    logger.info('Computing negative similarities...')
    neg_dists = cosine_indexed(feats, neg_pairs, batch_size).astype(np.float32)

    logger.info('Computing mAP and p-values per label...')
    negs_for = create_neg_query_solver(neg_pairs, neg_dists)
    ap_scores_list, p_values_list, ix_list = build_rank_lists_multi(
            pos_pairs, pos_dists, pos_counts, negs_for, null_size, seed
    )

    logger.info('Creating result DataFrame...')
    results = []
    for i, key in enumerate(pos_keys):
        result = pd.DataFrame({
            'average_precision': ap_scores_list[i],
            'p_value': p_values_list[i],
            'ix': ix_list[i],
            })
        if isinstance(key, tuple):
            # Is a ComposedKey
            for k, v in zip(key._fields, key):
                result[k] = v
        else:
            result[multilabel_col] = key
        results.append(result)
    results = pd.concat(results).reset_index(drop=True)
    meta = meta.drop(multilabel_col, axis=1)
    results = meta.merge(results, right_on='ix', left_index=True).drop('ix', axis=1)
    logger.info('Finished.')
    return results
