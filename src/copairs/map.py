from functools import partial
import logging

import numpy as np
import pandas as pd

from copairs.compute import compute_similarities
import copairs.compute_np as backend
from copairs.matching import Matcher

logger = logging.getLogger('copairs')


def build_rank_lists(pos_dfs, pos_sameby, neg_dfs, neg_sameby) -> pd.Series:
    pos_ids = pos_dfs.melt(value_vars=['ix1', 'ix2'],
                           id_vars=['dist'],
                           value_name='ix')
    pos_ids['label'] = 1
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


def find_pairs(obs: pd.DataFrame, pos_sameby, pos_diffby, neg_sameby,
               neg_diffby):
    matcher = Matcher(obs, obs.columns, seed=0)
    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = np.vstack(list(dict_pairs.values()))
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = np.vstack(list(dict_pairs.values()))
    return pos_pairs, neg_pairs


def results_to_dframe(meta, p_values, null_dists, aps):
    result = meta.copy().astype(str)
    result['p_value'] = p_values
    result['null_dists'] = list(null_dists)
    result['average_precision'] = aps
    return result


def run_pipeline(meta,
                 feats,
                 pos_sameby,
                 pos_diffby,
                 neg_sameby,
                 neg_diffby,
                 null_size,
                 batch_size=20000) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()
    logger.info('Finding positive and negative pairs...')
    pos_pairs, neg_pairs = find_pairs(meta, pos_sameby, pos_diffby, neg_sameby,
                                      neg_diffby)
    logger.info('Computing positive similarities...')
    pos_dfs = compute_similarities(feats, pos_pairs, batch_size)
    logger.info('Computing negative similarities...')
    neg_dfs = compute_similarities(feats, neg_pairs, batch_size)
    logger.info('Building rank lists...')
    rel_k_list = build_rank_lists(pos_dfs, pos_sameby, neg_dfs, neg_sameby)
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
