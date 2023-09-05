import itertools
import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs.compute import cosine_indexed
from copairs import compute_np
from copairs.matching import Matcher, MatcherMultilabel

logger = logging.getLogger('copairs')


def get_rel_k_list(row):
    num_pos = len(row['pos_dist'])
    num_neg = len(row['neg_dist'])
    label = np.repeat([1, 0], [num_pos, num_neg])
    dist = np.concatenate([row['pos_dist'], row['neg_dist']])
    return np.expand_dims(label[np.argsort(dist)[::-1]], axis=0)


def build_rank_list_multi(pos_dfs, neg_dfs, multilabel_col) -> pd.Series:
    '''Build a rank list for every (index, label) pair.'''
    pos_dfs = pos_dfs[[multilabel_col, 'ix1', 'ix2', 'dist']]
    pos_ids = pos_dfs.melt(value_vars=['ix1', 'ix2'],
                           id_vars=[multilabel_col, 'dist'],
                           value_name='ix')

    neg_dfs = neg_dfs[['ix1', 'ix2', 'dist']]
    neg_ids = neg_dfs.melt(value_vars=['ix1', 'ix2'],
                           id_vars=['dist'],
                           value_name='ix')

    dists = pos_ids.groupby([multilabel_col, 'ix'])['dist'].apply(list)
    dists.name = 'pos_dist'
    dists = dists.reset_index()

    neg_ids = neg_ids.groupby('ix')['dist'].apply(list)
    dists['neg_dist'] = dists['ix'].map(neg_ids)
    del pos_ids, neg_ids

    dists['rel_k_list'] = dists.apply(get_rel_k_list, axis=1)
    rel_k_list = dists.set_index([multilabel_col, 'ix']).rel_k_list

    return rel_k_list


def build_rank_lists(pos_pairs, neg_pairs, pos_dists, neg_dists):
    labels = np.concatenate([np.ones(pos_pairs.size, dtype=np.int32),
                             np.zeros(neg_pairs.size, dtype=np.int32)])
    ix = np.concatenate([pos_pairs.ravel(), neg_pairs.ravel()])
    # del pos_pairs, neg_pairs
    dist_all = np.concatenate([np.repeat(pos_dists, 2), np.repeat(neg_dists, 2)])
    # del pos_dists, neg_dists
    ix_sort = np.lexsort([1 - dist_all, ix])
    rel_k_list = labels[ix_sort]
    _, counts = np.unique(ix, return_counts=True)
    return rel_k_list, counts


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


def compute_similarities(pairs, feats, batch_size):
    dist_df = pairs[['ix1', 'ix2']].drop_duplicates().copy()
    dist_df['dist'] = cosine_indexed(feats, dist_df.values, batch_size)
    return pairs.merge(dist_df, on=['ix1', 'ix2'])


def results_to_dframe(meta, index, p_values, ap_scores, multilabel_col):
    scores = pd.DataFrame({'average_precision': ap_scores, 'p_value': p_values},
                          index=index)
    if multilabel_col is None or multilabel_col not in scores.index.names:
        result = meta.join(scores)
        return result
    meta = meta.drop(multilabel_col, axis=1)
    scores.reset_index(inplace=True)
    result = meta.merge(scores, left_index=True, right_on='ix')
    return result.drop('ix', axis=1)


def run_pipeline(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    null_size,
    multilabel_col=None,
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
    total_pos = sum(len(p) for p in pos_pairs.values())
    pos_pairs = np.fromiter(itertools.chain.from_iterable(pos_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=total_pos)

    logger.info('Finding negative pairs...')
    neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    total_neg = sum(len(p) for p in neg_pairs.values())
    neg_pairs = np.fromiter(itertools.chain.from_iterable(neg_pairs.values()),
                            dtype=np.dtype((np.int32, 2)),
                            count=total_neg)

    logger.info('Computing positive similarities...')
    pos_dists = cosine_indexed(feats, pos_pairs, batch_size)
    logger.info('Computing negative similarities...')
    neg_dists = cosine_indexed(feats, neg_pairs, batch_size)

    logger.info('Building rank lists...')
    rel_k_list, counts = build_rank_lists(pos_pairs, neg_pairs, pos_dists, neg_dists)
    logger.info('Computing average precision...')
    ap_scores, null_confs = compute_np.compute_ap_contiguos(rel_k_list, counts)

    logger.info('Computing P-values...')
    p_values = compute_np.compute_p_values(ap_scores, null_confs, null_size, seed=seed)
    logger.info('Creating result DataFrame...')
    result = results_to_dframe(meta, meta.index, p_values, ap_scores,
                               multilabel_col)
    logger.info('Finished.')
    return result


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
    return agg_rs
