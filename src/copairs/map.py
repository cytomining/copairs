from functools import partial
import itertools
import logging
import re

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from copairs.compute import cosine_indexed
import copairs.compute_np as backend
from copairs.matching import Matcher, MatcherMultilabel, dict_to_dframe

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


def evaluate_and_filter(df, columns) -> list:
    '''Evaluate the query and filter the dataframe'''
    parsed_cols = []
    for col in columns:
        if col in df.columns:
            parsed_cols.append(col)
            continue

        column_names = re.findall(r'(\w+)\s*[=<>!]+', col)
        valid_column_names = [col for col in column_names if col in df.columns]
        if not valid_column_names:
            raise ValueError(f"Invalid query or column name: {col}")

        try:
            df = df.query(col)
            parsed_cols.extend(valid_column_names)
        except:
            raise ValueError(f"Invalid query expression: {col}")

    return df, parsed_cols


def flatten_str_list(*args):
    '''create a single list with all the params given'''
    columns = set()
    for col in args:
        if isinstance(col, str):
            columns.add(col)
        elif isinstance(col, dict):
            columns.update(itertools.chain.from_iterable(col.values()))
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
    obs, columns = evaluate_and_filter(obs, columns)
    if multilabel_col:
        return MatcherMultilabel(obs, columns, multilabel_col, seed=0)
    return Matcher(obs, columns, seed=0)


def compute_similarities(pairs, feats, batch_size):
    dist_df = pairs[['ix1', 'ix2']].drop_duplicates().copy()
    dist_df['dist'] = cosine_indexed(feats, dist_df.values, batch_size)
    return pairs.merge(dist_df, on=['ix1', 'ix2'])


def results_to_dframe(meta, index, p_values, ap_scores, multilabel_col):
    scores = pd.DataFrame({
        'p_value': p_values,
        'average_precision': ap_scores
    },
                          index=index)
    if multilabel_col not in scores.index.names:
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
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    logger.info('Indexing metadata...')
    matcher = create_matcher(meta, pos_sameby, pos_diffby, neg_sameby,
                             neg_diffby, multilabel_col)

    logger.info('Finding positive pairs...')
    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    logger.info('dropping dups...')
    pos_pairs = dict_to_dframe(dict_pairs, pos_sameby)
    logger.info('Finding negative pairs...')
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    logger.info('dropping dups...')
    neg_pairs = set(itertools.chain.from_iterable(dict_pairs.values()))
    neg_pairs = pd.DataFrame(neg_pairs, columns=['ix1', 'ix2'])
    logger.info('Computing positive similarities...')
    pos_pairs = compute_similarities(pos_pairs, feats, batch_size)
    logger.info('Computing negative similarities...')
    neg_pairs = compute_similarities(neg_pairs, feats, batch_size)
    logger.info('Building rank lists...')
    if multilabel_col and multilabel_col in pos_sameby:
        rel_k_list = build_rank_list_multi(pos_pairs, neg_pairs,
                                           multilabel_col)
    else:
        rel_k_list = build_rank_lists(pos_pairs, neg_pairs)
    logger.info('Computing average precision...')
    ap_scores = rel_k_list.apply(backend.compute_ap)
    ap_scores = np.concatenate(ap_scores.values)
    logger.info('Computing null distributions...')
    null_dists = backend.compute_null_dists(rel_k_list, null_size)
    logger.info('Computing P-values...')
    p_values = backend.compute_p_values(null_dists, ap_scores, null_size)
    logger.info('Creating result DataFrame...')
    result = results_to_dframe(meta, rel_k_list.index, p_values, ap_scores,
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
