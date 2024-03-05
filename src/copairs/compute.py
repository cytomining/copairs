import itertools
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm.autonotebook import tqdm


def parallel_map(par_func, items):
    """Execute par_func(i) for every i in items using ThreadPool and tqdm."""
    num_items = len(items)
    pool_size = min(num_items, os.cpu_count())
    chunksize = num_items // pool_size
    with ThreadPool(pool_size) as pool:
        tasks = pool.imap_unordered(par_func, items, chunksize=chunksize)
        for _ in tqdm(tasks, total=len(items), leave=False):
            pass


def batch_processing(
    pairwise_op: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    """Decorator adding the batch_size param to run the function with
    multithreading using a list of paired indices"""

    def batched_fn(feats: np.ndarray, pair_ix: np.ndarray, batch_size: int):
        num_pairs = len(pair_ix)
        result = np.empty(num_pairs, dtype=np.float32)

        def par_func(i):
            x_sample = feats[pair_ix[i : i + batch_size, 0]]
            y_sample = feats[pair_ix[i : i + batch_size, 1]]
            result[i : i + len(x_sample)] = pairwise_op(x_sample, y_sample)

        parallel_map(par_func, np.arange(0, num_pairs, batch_size))

        return result

    return batched_fn


@batch_processing
def pairwise_corr(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """
    Compute pearson correlation between two matrices in a paired row-wise
    fashion. `x_sample` and `y_sample` must be of the same shape.
    """
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)

    x_center = x_sample - x_mean
    y_center = y_sample - y_mean

    numer = (x_center * y_center).sum(axis=1)

    denom = (x_center**2).sum(axis=1) * (y_center**2).sum(axis=1)
    denom = np.sqrt(denom)

    corrs = numer / denom
    return corrs


@batch_processing
def pairwise_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    x_norm = x_sample / np.linalg.norm(x_sample, axis=1)[:, np.newaxis]
    y_norm = y_sample / np.linalg.norm(y_sample, axis=1)[:, np.newaxis]
    c_sim = np.sum(x_norm * y_norm, axis=1)
    return c_sim


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


def average_precision(rel_k) -> np.ndarray:
    """Compute average precision based on binary list sorted by relevance"""
    tp = np.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = np.arange(1, rel_k.shape[1] + 1)
    pr_k = tp / k
    ap = (pr_k * rel_k).sum(axis=1) / num_pos
    return ap


def ap_contiguous(rel_k_list, counts):
    """Compute average precision from a list of contiguous values"""
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


def random_ap(num_perm: int, num_pos: int, total: int, seed) -> np.ndarray:
    """Compute multiple average_precision scores generated at random"""
    rng = np.random.default_rng(seed)
    rel_k = random_binary_matrix(num_perm, total, num_pos, rng)
    null_dist = average_precision(rel_k)
    return null_dist


def null_dist_cached(num_pos, total, seed, null_size, cache_dir):
    if seed is not None:
        cache_file = cache_dir / f"n{total}_k{num_pos}.npy"
        if cache_file.is_file():
            null_dist = np.load(cache_file)
        else:
            null_dist = random_ap(null_size, num_pos, total, seed)
            np.save(cache_file, null_dist)
    else:
        null_dist = random_ap(null_size, num_pos, total, seed)
    return null_dist


def get_null_dists(confs, null_size, seed):
    cache_dir = Path.home() / ".copairs" / f"seed{seed}" / f"ns{null_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    num_confs = len(confs)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(8096, size=num_confs)

    null_dists = np.empty([len(confs), null_size], dtype=np.float32)

    def par_func(i):
        num_pos, total = confs[i]
        null_dists[i] = null_dist_cached(num_pos, total, seeds[i], null_size, cache_dir)

    parallel_map(par_func, np.arange(num_confs))
    return null_dists


def p_values(ap_scores: np.ndarray, null_confs: np.ndarray, null_size: int, seed: int):
    """Calculate p values for an array of ap_scores and null configurations. It uses the path
    folder to cache null calculations.

    Parameters
    ----------
    ap_scores : np.ndarray
        Ap scores for which to calculate p value.
    null_confs : np.ndarray
        Number of average precisions calculated. It serves as an indicator of
        how relevant is the resultant score.
    null_size : int
    seed : int
        Random initializing value.

    Examples
    --------
    FIXME: Add docs.


    """
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = get_null_dists(confs, null_size, seed)
    null_dists.sort(axis=1)
    pvals = np.empty(len(ap_scores), dtype=np.float32)
    for i, (ap_score, ix) in enumerate(zip(ap_scores, rev_ix)):
        # Reverse to get from hi to low
        num = null_size - np.searchsorted(null_dists[ix], ap_score)
        pvals[i] = (num + 1) / (null_size + 1)
    return pvals


def concat_ranges(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Create a 1-d array concatenating multiple ranges"""
    slices = map(range, start, end)
    slices = itertools.chain.from_iterable(slices)
    count = (end - start).sum()
    mask = np.fromiter(slices, dtype=np.int32, count=count)
    return mask


def to_cutoffs(counts: np.ndarray):
    """Convert a list of counts into cutoff indices."""
    cutoffs = np.empty_like(counts)
    cutoffs[0], cutoffs[1:] = 0, counts.cumsum()[:-1]
    return cutoffs
