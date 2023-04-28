from functools import partial, lru_cache
from typing import Callable
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

NUM_PROC = 4


def process_batch(i: int, batch_size: int, feats: np.ndarray,
                  pair_ix: np.ndarray, batch_pairwise_op):
    x_sample = feats[pair_ix[i:i + batch_size, 0]]
    y_sample = feats[pair_ix[i:i + batch_size, 1]]
    corr = batch_pairwise_op(x_sample, y_sample)
    return corr


def pairwise_indexed(feats: np.ndarray, pair_ix: np.ndarray,
                     batch_pairwise_op: Callable[[np.ndarray, np.ndarray],
                                                 np.ndarray], batch_size):
    '''Get pairwise correlation using a list of paired indices'''
    num_pairs = len(pair_ix)

    corrs = []

    par_func = partial(process_batch,
                       batch_size=batch_size,
                       feats=feats,
                       pair_ix=pair_ix,
                       batch_pairwise_op=batch_pairwise_op)
    with Pool(NUM_PROC) as p:
        idx = list(range(0, num_pairs, batch_size))
        corrs = list(tqdm(p.imap(par_func, idx), total=len(idx), leave=False))

    corrs = np.concatenate(corrs)
    assert len(corrs) == num_pairs
    return corrs


def pairwise_corr(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlation between two matrices in a paired row-wise
    fashion. `x_sample` and `y_sample` must be of the same shape.
    '''
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)

    x_center = x_sample - x_mean
    y_center = y_sample - y_mean

    numer = (x_center * y_center).sum(axis=1)

    denom = (x_center**2).sum(axis=1) * (y_center**2).sum(axis=1)
    denom = np.sqrt(denom)

    corrs = numer / denom
    return corrs


def pairwise_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    x_norm = x_sample / np.linalg.norm(x_sample, axis=1)[:, np.newaxis]
    y_norm = y_sample / np.linalg.norm(y_sample, axis=1)[:, np.newaxis]
    c_dist = np.sum(x_norm * y_norm, axis=1)
    return c_dist


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


def compute_null_dists(rel_k_list, null_size):
    num_pos_list = rel_k_list.apply(np.sum)
    num_neg_list = rel_k_list.apply(np.size) - num_pos_list
    null_confs = []
    for num_pos, num_neg in zip(num_pos_list, num_neg_list):
        key = null_size, num_pos, num_neg
        null_confs.append(key)
    with Pool(processes=NUM_PROC) as pool:
        null_dists = np.stack(pool.starmap(random_ap, null_confs))
    # null_dists = np.stack([random_ap(*key) for key in null_confs])
    return null_dists


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
