import itertools
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

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
    matrix = np.zeros((n, m), dtype=int)
    matrix[:, :k] = 1
    rng.permuted(matrix, axis=1, out=matrix)
    return matrix


def compute_ap(rel_k) -> np.ndarray:
    '''Compute average precision based on binary list sorted by relevance'''
    tp = np.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = np.arange(1, rel_k.shape[1] + 1)
    pr_k = tp / k
    ap = (pr_k * rel_k).sum(axis=1) / num_pos
    return ap


def compute_ap_contiguos(rel_k_list, counts):
    '''Compute average precision from a list of contiguous values'''
    cutoffs = to_cutoffs(counts)

    num_pos = np.add.reduceat(rel_k_list, cutoffs)
    shift = np.empty_like(num_pos)
    shift[0], shift[1:] = 0, num_pos[:-1]

    tp = rel_k_list.cumsum() - np.repeat(shift.cumsum(), counts)
    k = np.arange(1, len(rel_k_list) + 1) - np.repeat(cutoffs, counts)

    pr_k = tp / k
    ap_scores = np.add.reduceat(pr_k * rel_k_list, cutoffs) / num_pos
    null_confs = np.stack([num_pos, counts], axis=1)
    return ap_scores, null_confs


def _random_ap(num_perm: int, num_pos: int, total: int, seed) -> np.ndarray:
    '''Compute multiple average_precision scores generated at random'''
    rng = np.random.default_rng(seed)
    rel_k = random_binary_matrix(num_perm, total, num_pos, rng)
    return compute_ap(rel_k)


def null_dist_cached(total, num_pos, null_size, seed, cache_dir):
    if seed is not None:
        cache_file = cache_dir / f'n{total}_k{num_pos}.npy'
        if cache_file.is_file():
            null_dist = np.load(cache_file)
        else:
            null_dist = _random_ap(null_size, num_pos, total, seed)
            null_dist.sort()
            np.save(cache_file, null_dist)
    else:
        null_dist = _random_ap(null_size, num_pos, total, seed)
    return null_dist


def get_null_dists(confs, null_size, seed):
    cache_dir = Path.home() / f'.copairs/seed{seed}/ns{null_size}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    par_func = partial(null_dist_cached,
                       cache_dir=cache_dir,
                       null_size=null_size)
    null_dists = np.empty([len(confs), null_size])
    rng = np.random.default_rng(seed)
    seeds = rng.integers(8096, size=len(confs))
    for i, (num_pos, total) in enumerate(tqdm(confs, leave=False)):
        null_dists[i] = par_func(total, num_pos, seed=seeds[i])
    return null_dists


def compute_p_values(ap_scores, null_confs, null_size: int, seed):
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = get_null_dists(confs, null_size, seed)
    p_values = np.empty(len(ap_scores), dtype=np.float32)
    for i, (ap_score, ix) in enumerate(zip(ap_scores, rev_ix)):
        # Reverse to get from hi to low
        num = null_size - np.searchsorted(null_dists[ix], ap_score)
        p_values[i] = (num + 1) / (null_size + 1)
    return p_values


def concat_ranges(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    '''Create a 1-d array concatenating multiple ranges'''
    slices = map(range, start, end)
    slices = itertools.chain.from_iterable(slices)
    count = (end - start).sum()
    mask = np.fromiter(slices, dtype=np.int32, count=count)
    return mask


def to_cutoffs(counts: np.ndarray):
    '''Convert a list of counts into cutoff indices.'''
    cutoffs = np.empty_like(counts)
    cutoffs[0], cutoffs[1:] = 0, counts.cumsum()[:-1]
    return cutoffs
