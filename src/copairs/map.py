from functools import partial
import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs.compute import cosine_indexed
import copairs.compute_np as backend
from copairs.matching import Matcher, MatcherMultilabel, dict_to_dframe

logger = logging.getLogger('copairs')


def build_rank_lists(pos_dfs, neg_dfs) -> pd.Series:
    pos_dfs = pos_dfs[['ix1', 'ix2', 'dist']]
    pos_ids = pos_dfs.melt(value_vars=['ix1', 'ix2'],
                           id_vars=['dist'],
                           value_name='ix')
    pos_ids['label'] = 1

    neg_dfs = neg_dfs[['ix1', 'ix2', 'dist']]
    neg_ids = neg_dfs.melt(value_vars=['ix1', 'ix2'],
                           id_vars=['dist'],
                           value_name='ix')
    neg_ids['label'] = 0
    dists = pd.concat([pos_ids, neg_ids])
    del pos_ids, neg_ids
    dists = dists.sort_values(['ix', 'dist'], ascending=[True, False])
    rel_k_list = dists.groupby('ix', sort=False)['label'].apply(
        partial(np.expand_dims, axis=0))
    return rel_k_list


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


def results_to_dframe(meta, p_values, null_dists, aps):
    result = meta.copy().astype(str)
    result['p_value'] = p_values
    result['null_dists'] = list(null_dists)
    result['average_precision'] = aps
    return result


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
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info('Indexing metadata...')
    matcher = create_matcher(meta, pos_sameby, pos_diffby, neg_sameby,
                             neg_diffby, multilabel_col)
    logger.info('Finding positive pairs...')
    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = dict_to_dframe(dict_pairs, pos_sameby)
    logger.info('Finding negative pairs...')
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = dict_to_dframe(dict_pairs, neg_sameby)
    logger.info('Computing positive similarities...')
    pairs_ix = pos_pairs[['ix1', 'ix2']].values
    pos_pairs['dist'] = cosine_indexed(feats, pairs_ix, batch_size)
    logger.info('Computing negative similarities...')
    pairs_ix = neg_pairs[['ix1', 'ix2']].values
    neg_pairs['dist'] = cosine_indexed(feats, pairs_ix, batch_size)
    logger.info('Building rank lists...')
    rel_k_list = build_rank_lists(pos_pairs, neg_pairs)
    logger.info('Computing average precision...')
    ap_scores = rel_k_list.apply(backend.compute_ap)
    ap_scores = np.concatenate(ap_scores.values)
    logger.info('Computing null distributions...')
    null_dists = backend.compute_null_dists(rel_k_list, null_size)
    logger.info('Computing P-values...')
    p_values = backend.compute_p_values(null_dists, ap_scores, null_size)
    logger.info('Creating result DataFrame...')
    result = results_to_dframe(meta, p_values, null_dists, ap_scores)
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
