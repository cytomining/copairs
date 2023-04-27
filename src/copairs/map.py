import logging
from typing import Tuple
import pandas as pd
import numpy as np
from functools import partial, lru_cache
import multiprocessing
from copairs.compute_np import NUM_PROC
from copairs.matching import Matcher
from copairs.compute import compute_similarities

logger = logging.getLogger('copairs')


def random_binary_matrix(n, m, k, rng):
    """Generate a random binary matrix of n*m with exactly k values in 1 per row.
    Args:
    n: Number of rows.
    m: Number of columns.
    k: Number of 1's per row.

    Returns:
    A: Random binary matrix of n*m with exactly k values in 1 per row.
    """

    # Initialize the matrix.
    matrix = np.zeros((n, m), dtype=int)
    matrix[:, :k] = 1

    # Shuffle inplace
    np.apply_along_axis(rng.shuffle, axis=1, arr=matrix)
    return matrix


def compute_ap(rel_k) -> np.ndarray:
    '''Compute average precision based on binary list sorted by relevance'''
    tp = np.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = np.arange(1, rel_k.shape[1] + 1)
    pr_k = tp / k
    ap = (pr_k * rel_k).sum(axis=1) / num_pos
    return ap


@lru_cache(maxsize=None)
def random_ap(num_perm: int,
              num_pos: int,
              num_neg: int,
              seed=None) -> np.ndarray:
    '''Compute multiple average_precision scores generated at random'''
    total = num_pos + num_neg
    rng = np.random.default_rng(seed)
    rel_k = random_binary_matrix(num_perm, total, num_pos, rng)
    return compute_ap(rel_k)


def compute_p_values(null_dists, ap_scores, null_size):
    '''Compute p-values'''
    num = 1 + (null_dists > ap_scores[:, None]).sum(axis=1)
    denom = 1 + null_size
    p_values = num / denom
    return p_values


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


def compute_null_dists(rel_k_list, null_size):
    num_pos_list = rel_k_list.apply(np.sum)
    num_neg_list = rel_k_list.apply(np.size) - num_pos_list
    null_confs = []
    for num_pos, num_neg in zip(num_pos_list, num_neg_list):
        key = null_size, num_pos, num_neg
        null_confs.append(key)
    with multiprocessing.Pool(processes=NUM_PROC) as pool:
        null_dists = np.stack(pool.starmap(random_ap, null_confs))
    # null_dists = np.stack([random_ap(*key) for key in null_confs])
    return null_dists, null_size


def find_pairs(obs: pd.DataFrame, pos_sameby, pos_diffby, neg_sameby,
               neg_diffby):
    matcher = Matcher(obs, obs.columns, seed=0)
    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = np.vstack(list(dict_pairs.values()))
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = np.vstack(list(dict_pairs.values()))
    return pos_pairs, neg_pairs


def remove_unpaired(pos_pairs, neg_dfs, meta,
                    pos_sameby) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Remove indices that do not have positive pairs in meta and negative distances'''
    if isinstance(pos_sameby, str):
        found_keys = pd.DataFrame({pos_sameby: pos_pairs.keys()})
    else:
        found_keys = pd.DataFrame(pos_pairs.keys())
    found_index = meta.merge(found_keys, on=pos_sameby).index
    miss_jcpid_ix = meta.index.difference(found_index)
    
    if len(miss_jcpid_ix) > 0:
        missing_jcpids = meta.loc[miss_jcpid_ix, pos_sameby].drop_duplicates()
        logger.warning(f'Can\'t find a valid pair for:\n{missing_jcpids}')
        query = 'ix1 not in @miss_jcpid_ix and ix2 not in @miss_jcpid_ix'
        neg_dfs = neg_dfs.query(query)

    meta_filtered = meta.drop(miss_jcpid_ix)
    return meta_filtered, neg_dfs


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
    logger.info('Removing unpaired samples...')
    # meta_filtered, neg_dfs = remove_unpaired(pos_pairs, neg_dfs, meta,
    #                                         pos_sameby)
    logger.info('Building rank lists...')
    rel_k_list = build_rank_lists(pos_dfs, pos_sameby, neg_dfs, neg_sameby)
    logger.info('Computing average precision...')
    ap_scores = rel_k_list.apply(compute_ap)
    ap_scores = np.concatenate(ap_scores.values)
    logger.info('Computing null distributions...')
    null_dists, null_size = compute_null_dists(rel_k_list, null_size)
    logger.info('Computing P-values...')
    
    p_values = compute_p_values(null_dists, ap_scores, null_size)
    logger.info('Creating result DataFrame...')
    result = results_to_dframe(meta, p_values, null_dists, ap_scores)
    logger.info('Finished.')
    return result
