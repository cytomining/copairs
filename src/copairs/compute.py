"""Functions to compute distances and ranks using numpy operations."""

import os
import tempfile
import itertools
from typing import Tuple, Union, Callable, Optional
from pathlib import Path
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.spatial.distance import _METRICS_NAMES as SCIPY_METRICS_NAMES
from scipy.spatial.distance import cdist

from copairs import nulls
from copairs.nulls.common import resolve_seed, check_null_size
from copairs.nulls.backend import check_method


def parallel_map(
    par_func: Callable[[int], None],
    items: np.ndarray,
    progress_bar: bool = True,
) -> None:
    """Execute a function in parallel over a list of items.

    This function uses a thread pool to process items in parallel, with progress
    tracking via `tqdm`. It is particularly useful for batch operations that benefit
    from multithreading.

    Parameters
    ----------
    par_func : Callable
        A function to execute for each item. It should accept a single argument
        (an item index or value).
    items : np.ndarray
        An array or list of items to process.
    """
    # Total number of items to process
    num_items = len(items)
    if num_items == 0:
        return

    # Determine the number of threads to use, limited by CPU count
    pool_size = min(num_items, os.cpu_count() or 1)

    # Calculate chunk size for dividing work among threads
    chunksize = num_items // pool_size

    # Use a thread pool to execute the function in parallel
    with ThreadPool(pool_size) as pool:
        # Map the function to items with unordered execution for better efficiency
        tasks = pool.imap_unordered(par_func, items, chunksize=chunksize)

        if progress_bar:
            # Display progress using tqdm
            from tqdm.autonotebook import tqdm

            tasks = tqdm(tasks, total=len(items), leave=False)
        for _ in tasks:
            pass


def batch_processing(
    pairwise_op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    progress_bar: bool = True,
):
    """
    Add batch processing support to pairwise operations.

    This decorator wraps a pairwise operation to process data in batches,
    enabling efficient computation and multithreading when working with large
    datasets.

    Parameters
    ----------
    pairwise_op : Callable
        A function that computes pairwise operations (e.g., similarity or distance)
        between two arrays of features.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.

    Returns
    -------
    Callable
        A wrapped function that processes pairwise operations in batches.

    """

    def batched_fn(feats: np.ndarray, pair_ix: np.ndarray, batch_size: int):
        # Total number of pairs to process
        num_pairs = len(pair_ix)

        # Initialize an empty result array to store pairwise operation results
        result = np.empty(num_pairs, dtype=np.float32)

        def par_func(i):
            # Extract the features for the current batch of pairs
            x_sample = feats[pair_ix[i : i + batch_size, 0]]
            y_sample = feats[pair_ix[i : i + batch_size, 1]]

            # Compute pairwise operations for the current batch
            result[i : i + len(x_sample)] = pairwise_op(x_sample, y_sample)

        # Use multithreading to process the batches in parallel
        parallel_map(
            par_func, np.arange(0, num_pairs, batch_size), progress_bar=progress_bar
        )

        return result

    return batched_fn


def pairwise_corr(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the Pearson correlation coefficient for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        A 1D array of Pearson correlation coefficients for each row pair in
        `x_sample` and `y_sample`.
    """
    # Compute the mean for each row
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)

    # Center the rows by subtracting the mean
    x_center = x_sample - x_mean
    y_center = y_sample - y_mean

    # Compute the numerator (dot product of centered vectors)
    numer = (x_center * y_center).sum(axis=1)

    # Compute the denominator (product of vector magnitudes)
    denom = (x_center**2).sum(axis=1) * (y_center**2).sum(axis=1)
    denom = np.sqrt(denom)

    # Calculate correlation coefficients
    corrs = numer / denom
    return corrs


def pairwise_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        A 1D array of cosine similarity scores for each row pair in `x_sample` and `y_sample`.
    """
    # Normalize each row to unit vectors
    x_norm = x_sample / np.linalg.norm(x_sample, axis=1)[:, np.newaxis]
    y_norm = y_sample / np.linalg.norm(y_sample, axis=1)[:, np.newaxis]

    # Compute the dot product of normalized vectors
    c_sim = np.sum(x_norm * y_norm, axis=1)
    return c_sim


def pairwise_abs_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the absolute cosine similarity for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        Absolute values of cosine similarity scores.
    """
    return np.abs(pairwise_cosine(x_sample, y_sample))


def pairwise_euclidean(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Euclidean distance for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        A 1D array of inverse Euclidean distance scores (scaled to range 0-1).
    """
    # Compute Euclidean distance and scale to a range of 0 to 1
    e_dist = np.sqrt(np.sum((x_sample - y_sample) ** 2, axis=1))
    return 1 / (1 + e_dist)


def pairwise_manhattan(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the inverse Manhattan distance for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        A 1D array of inverse Manhattan distance scores (scaled to range 0-1).
    """
    m_dist = np.sum(np.abs(x_sample - y_sample), axis=1)
    return 1 / (1 + m_dist)


def pairwise_chebyshev(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the inverse Chebyshev distance for paired rows of two matrices.

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns
    -------
    np.ndarray
        A 1D array of inverse Chebyshev distance scores (scaled to range 0-1).
    """
    c_dist = np.max(np.abs(x_sample - y_sample), axis=1)
    return 1 / (1 + c_dist)


def _cdist_diag_sim(
    x_sample: np.ndarray, y_sample: np.ndarray, metric: str
) -> np.ndarray:
    """Compute similarity based on the diagonal of the ScipY's cdist result (row-wise distance).

    Parameters
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.
    metric : str
        The name of the distance metric to use.

    Returns
    -------
    np.ndarray
        A 1D array of distance scores for each row pair in `x_sample` and `y_sample (diagonal).
    """
    bounded_0_1 = ["jaccard", "hamming"]
    distance = np.diag(cdist(x_sample, y_sample, metric=metric))
    if metric in bounded_0_1:
        return 1 - distance
    return 1 / (1 + distance)


def get_similarity_fn(
    distance: Union[str, Callable], progress_bar: bool = True
) -> Callable:
    """Retrieve a similarity function based on a distance string identifier or custom callable.

    This function provides flexibility in specifying the distance function to be used
    for pairwise similarity computations. Users can choose a metrics from a predefined set,
    scipy.spational.distance submodule, or provide a custom callable.

    Parameters
    ----------
    distance : str or callable
        The name of the distance function or a custom callable function. Supported
        string identifiers for predefined metrics are:
        - "cosine": Cosine similarity.
        - "abs_cosine": Absolute cosine similarity.
        - "correlation": Pearson correlation coefficient.
        - "euclidean": Inverse Euclidean distance (scaled to range 0-1).
        - "manhattan": Inverse Manhattan distance (scaled to range 0-1).
        - "chebyshev": Inverse Chebyshev distance (scaled to range 0-1).

        Additionally, any distance metric supported by `scipy.spatial.distance.cdist`
        can be used by providing the metric name as a string.

        If a callable is provided, it must accept the paramters associated with each
        callable function.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.

    Returns
    -------
    callable
        A function implementing the specified similarity function.

    Raises
    ------
    ValueError:
        If the provided `distance` is not a recognized string identifier or a valid callable.

    Example:
    -------
    >>> distance_fn = get_distance_fn("cosine")
    >>> similarity_scores = distance_fn(x_sample, y_sample)
    """
    # Dictionary of supported similarity functions
    similarity_functions = {
        "abs_cosine": pairwise_abs_cosine,
        "cosine": pairwise_cosine,
        "correlation": pairwise_corr,
        "euclidean": pairwise_euclidean,
        "manhattan": pairwise_manhattan,
        "chebyshev": pairwise_chebyshev,
    }

    # If a string is provided, look up the corresponding function
    if isinstance(distance, str):
        if distance in similarity_functions:
            similarity_fn = similarity_functions[distance]
        elif distance in SCIPY_METRICS_NAMES:
            similarity_fn = lambda x_sample, y_sample: _cdist_diag_sim(
                x_sample, y_sample, distance
            )
        else:
            raise ValueError(
                f"Unsupported distance function: {distance}. Supported functions are: {set(similarity_functions.keys()) | set(SCIPY_METRICS_NAMES)}"
            )
    elif callable(distance):
        # If a callable is provided, use it directly
        similarity_fn = distance
    else:
        # Raise an error if neither a string nor a callable is provided
        raise ValueError("Distance must be either a string or a callable object.")

    # Wrap the distance function for efficient batch processing
    return batch_processing(similarity_fn, progress_bar=progress_bar)


def random_binary_matrix(n, m, k, rng):
    """Generate a indices of k values in 1 per row in a random binary n*m matrix.

    Args:
    n: Number of rows.
    m: Number of columns.
    k: Number of 1's per row.

    Returns
    -------
    np.ndarray
        A binary matrix of shape `(n, m)` with exactly `k` ones per row.
    """
    dtype = np.uint16 if m < 2**16 else np.uint32
    indices = np.tile(np.arange(m, dtype=dtype), (n, 1))
    rng.permuted(indices, axis=1, out=indices)
    return np.sort(indices[:, :k], axis=1)


def average_precision(rel_k) -> np.ndarray:
    """Compute average precision based on binary list indices."""
    num_pos = rel_k.shape[1]
    pr_k = np.arange(1, num_pos + 1, dtype=np.float32) / (rel_k + 1)
    ap_values = pr_k.sum(axis=1) / num_pos
    return ap_values.astype(np.float32)


def ap_contiguous(
    rel_k_list: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Average Precision (AP) scores from relevance labels.

    This function calculates Average Precision (AP) scores for each profile based on
    relevance labels and their associated counts. It also returns configurations
    indicating the number of positive and total pairs for each profile.

    Parameters
    ----------
    rel_k_list : np.ndarray
        Array of relevance labels (1 for positive pairs, 0 for negative pairs), sorted
        by descending similarity within profiles.
    counts : np.ndarray
        Array indicating how many times each profile appears in the rank list.

    Returns
    -------
    ap_scores : np.ndarray
        Array of Average Precision scores for each profile.
    null_confs : np.ndarray
        Array of configurations, where each row corresponds to:
        - Number of positive pairs (`num_pos`).
        - Total number of pairs (`counts`).
    """
    # Convert counts into cutoff indices to segment relevance labels
    cutoffs = to_cutoffs(counts)

    num_pos = np.add.reduceat(rel_k_list, cutoffs, dtype=np.uint32)
    shift = np.empty_like(num_pos)
    shift[0], shift[1:] = 0, num_pos[:-1]

    # Calculate cumulative true positives for each profile segment
    tp = rel_k_list.cumsum() - np.repeat(shift.cumsum(), counts)

    # Rank positions for each relevance label, adjusted by cutoff indices
    k = np.arange(1, len(rel_k_list) + 1) - np.repeat(cutoffs, counts)

    # Compute precision at each rank (precision = TP / rank)
    pr_k = tp / k

    # Calculate average precision scores for each profile
    with np.errstate(divide="ignore", invalid="ignore"):
        ap_scores = np.add.reduceat(pr_k * rel_k_list, cutoffs) / num_pos

    # Generate configurations (number of positive and total pairs)
    null_confs = np.stack([num_pos, counts], axis=1)

    return ap_scores, null_confs


def random_ap(num_perm: int, num_pos: int, total: int, seed: int):
    """Generate random Average Precision (AP) scores to create a null distribution.

    This function computes multiple Average Precision (AP) scores based on randomly
    generated binary relevance lists. It is useful for generating a null distribution
    to assess the significance of observed AP scores.

    Parameters
    ----------
    num_perm : int
        Number of random permutations (i.e., how many random relevance lists to generate).
    num_pos : int
        Number of positive samples (1's) in each relevance list.
    total : int
        Total number of samples (columns) in each relevance list.
    seed : int
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    np.ndarray
        A 1D array containing the Average Precision scores for each randomly
        generated relevance list.
    """
    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Generate a binary matrix with `num_perm` rows and `total` columns,
    # where each row contains exactly `num_pos` ones distributed randomly
    rel_k = random_binary_matrix(num_perm, total, num_pos, rng)

    # Compute Average Precision (AP) scores for each row of the binary matrix
    null_dist = average_precision(rel_k)
    return null_dist


def _cache_read(path: Path) -> Optional[np.ndarray]:
    """Read a cached .npy file, returning None on any corruption or absence."""
    try:
        return np.load(path)
    except (FileNotFoundError, ValueError, EOFError, OSError):
        return None


def _cache_write(path: Path, arr: np.ndarray) -> None:
    """Write a .npy file atomically via temp file + os.replace.

    os.replace is atomic on POSIX (same filesystem), so concurrent readers
    see either the old complete file or the new complete file, never a
    partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def null_dist_cached(
    num_pos: int, total: int, seed: int, null_size: int, cache_dir: Path
) -> np.ndarray:
    """Generate or retrieve a cached null distribution for a given configuration.

    Each (num_pos, total, seed) triple is cached as an individual .npy file,
    so a cached null is exactly the one ``random_ap`` would generate. Reads
    are lock-free (just np.load), and writes use atomic rename to prevent
    corruption from concurrent workers.

    Parameters
    ----------
    num_pos : int
        Number of positive pairs in the configuration.
    total : int
        Total number of pairs (positive + negative) in the configuration.
    seed : int
        Random seed for reproducibility.
    null_size : int
        Number of samples to generate in the null distribution.
    cache_dir : Path
        Directory to store or retrieve cached null distributions.

    Returns
    -------
    np.ndarray
        Null distribution for the specified configuration.
    """
    if seed is None:
        return random_ap(null_size, num_pos, total, seed)

    path = cache_dir / f"n{total}_k{num_pos}_s{seed}.npy"
    null_dist = _cache_read(path)
    if null_dist is None:
        null_dist = random_ap(null_size, num_pos, total, seed)
        _cache_write(path, null_dist)
    return null_dist


def _conf_seed(seed: int, num_pos: int, total: int) -> int:
    """Return the seed of one configuration's null, a function of its values only."""
    return int(np.random.SeedSequence([seed, num_pos, total]).generate_state(1)[0])


def get_null_dists(
    confs: np.ndarray,
    null_size: int,
    seed: int,
    cache_dir: Optional[Union[str, Path]] = None,
    progress_bar: bool = True,
    method: str = "fast",
    backend: str = "auto",
) -> np.ndarray:
    """Generate null distributions for each configuration of positive and total pairs.

    Parameters
    ----------
    confs : np.ndarray
        Array where each row contains the number of positive pairs (`num_pos`)
        and total pairs (`total`) for a specific configuration.
    null_size : int
        Number of samples to generate in the null distribution.
    seed : int
        Random seed for reproducibility. Each configuration's null is seeded by
        ``(seed, num_pos, total)``, so it does not depend on the other
        configurations. With ``None``, nulls are fresh draws and are not cached.
    cache_dir : str or Path, optional
        Cache location of ``method="legacy"`` (default ``~/.copairs``). The fast
        method regenerates nulls deterministically and does not cache them.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    method : str
        ``"fast"`` (default) samples each null exactly with
        :func:`copairs.nulls.ap_nulls`: sample ``j`` depends only on
        ``(seed, num_pos, total, j)`` (seed in ``[0, 2**64)``), in O(1) memory
        per sample. ``"legacy"`` permutes a ``null_size x total`` matrix per
        configuration and caches it, as copairs 0.5.5 did; its nulls are those
        of 0.5.5 within Monte Carlo error, not bitwise, since each
        configuration is now seeded by its values rather than its position.
    backend : str
        Backend of the fast method: ``"auto"`` (default), ``"numba"`` or
        ``"numpy"``. All return identical values, so ``"auto"`` picks
        ``"numba"`` whenever it is installed.

    Returns
    -------
    np.ndarray
        A 2D float32 array where each row corresponds to a null distribution for
        a specific configuration. Configurations without positives get NaN.
    """
    check_method(method, backend)
    if method == "fast":
        return _fast_null_dists(confs, null_size, seed, backend)
    cache_dir = Path.home() / ".copairs" if cache_dir is None else Path(cache_dir)
    cache_dir = cache_dir / f"seed{seed}" / f"ns{null_size}"
    if seed is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    num_confs = len(confs)
    null_dists = np.empty([len(confs), null_size], dtype=np.float32)

    def par_func(i):
        num_pos, total = confs[i]
        conf_seed = None if seed is None else _conf_seed(seed, num_pos, total)
        null_dists[i] = null_dist_cached(
            num_pos, total, conf_seed, null_size, cache_dir
        )

    parallel_map(par_func, np.arange(num_confs), progress_bar)

    return null_dists


def _without_positives(confs: np.ndarray) -> np.ndarray:
    """Mask of configurations with no positives, whose AP is undefined.

    The legacy method gives them NaN nulls and p-values of 1; sample ``j`` of
    any other configuration does not depend on them.
    """
    confs = np.asarray(confs).reshape(-1, 2)
    return (confs[:, 0] == 0) & (confs[:, 1] >= 0)


def _fast_null_dists(confs, null_size, seed, backend) -> np.ndarray:
    """:func:`get_null_dists` with ``method="fast"``."""
    confs = np.asarray(confs).reshape(-1, 2)
    null_size = check_null_size(null_size)
    out = np.full((len(confs), null_size), np.nan, dtype=np.float32)
    ranked = ~_without_positives(confs)
    if ranked.any():
        out[ranked] = nulls.ap_nulls(
            confs[ranked], null_size, resolve_seed(seed), backend=backend
        )
    return out


def _fast_p_values(ap_scores, confs, rev_ix, null_size, seed, backend, progress_bar):
    """:func:`p_values` with ``method="fast"``."""
    ranked = ~_without_positives(confs)
    pvals = np.ones(len(rev_ix), dtype=np.float32)
    rows = ranked[rev_ix]
    if not rows.any():
        check_null_size(null_size)
        nulls.resolve_backend(backend)
        return pvals
    index = np.cumsum(ranked) - 1  # configuration -> row of confs[ranked]
    pvals[rows] = nulls.ap_pvalues(
        np.asarray(ap_scores)[rows],
        index[rev_ix[rows]],
        confs[ranked],
        null_size,
        seed,
        backend=backend,
        progress_bar=progress_bar,
    )
    return pvals


def p_values(
    ap_scores: np.ndarray,
    null_confs: np.ndarray,
    null_size: int,
    seed: int,
    progress_bar: bool = True,
    method: str = "fast",
    backend: str = "auto",
):
    """Calculate p-values for an array of Average Precision (AP) scores using a null distribution.

    Parameters
    ----------
    ap_scores : np.ndarray
        Array of observed AP scores for which to calculate p-values.
    null_confs : np.ndarray
        Configuration array indicating the relevance or context of each AP score. Used
        to generate corresponding null distributions.
    null_size : int
        Number of samples to generate in the null distribution for each configuration.
    seed : int
        Seed for the random number generator to ensure reproducibility of the null
        distribution.
    progress_bar : bool
        Whether or not to show tqdm's progress bar.
    method : str
        ``"fast"`` (default) streams exact null samples (see
        :func:`get_null_dists`) in bounded memory, without a cache, and counts
        null scores ``>= score``, treating values within the scores' rounding
        as ties (see :func:`copairs.nulls.tie_thresholds`). ``"legacy"``
        materializes and caches the nulls of :func:`get_null_dists`'s legacy
        method and counts null scores ``>= score`` exactly.
    backend : str
        Backend of the fast method, see :func:`get_null_dists`.

    Returns
    -------
    np.ndarray
        An array of p-values corresponding to the input AP scores.
    """
    check_method(method, backend)
    # Identify unique configurations and their indices
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    if method == "fast":
        return _fast_p_values(
            ap_scores, confs, rev_ix.ravel(), null_size, seed, backend, progress_bar
        )

    # Generate null distributions for each unique configuration
    null_dists = get_null_dists(
        confs, null_size, seed, progress_bar=progress_bar, method="legacy"
    )

    # Sort null distributions for efficient p-value computation
    null_dists.sort(axis=1)

    # Initialize an array to store the p-values
    pvals = np.empty(len(ap_scores), dtype=np.float32)

    # Compute p-values for each AP score
    for i, (ap_score, ix) in enumerate(zip(ap_scores, rev_ix)):
        # Find the rank of the observed AP score in the sorted null distribution
        num = null_size - np.searchsorted(null_dists[ix], ap_score)

        # Calculate the p-value as the proportion of null scores >= observed score
        pvals[i] = (num + 1) / (null_size + 1)

    return pvals


def concat_ranges(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Create a 1D array by concatenating multiple integer ranges.

    This function generates a single concatenated array from multiple ranges defined
    by the `start` and `end` arrays. Each range is inclusive of `start` and exclusive
    of `end`.

    Parameters
    ----------
    start : np.ndarray
        A 1D array of start indices for the ranges.
    end : np.ndarray
        A 1D array of end indices for the ranges. Must have the same shape as `start`.

    Returns
    -------
    np.ndarray
        A 1D array containing the concatenated ranges.
    """
    # Generate individual ranges using `range` for each pair of start and end
    slices = map(range, start, end)

    # Flatten the ranges into a single iterable
    slices = itertools.chain.from_iterable(slices)

    # Calculate the total length of the concatenated ranges
    count = (end - start).sum()

    # Create a 1D array from the concatenated ranges
    mask = np.fromiter(slices, dtype=np.int32, count=count)

    return mask


def to_cutoffs(counts: np.ndarray) -> np.ndarray:
    """Convert counts into cumulative cutoff indices.

    This function generates a 1D array of indices that mark the start of each segment
    in a cumulative list. The first index is always `0`, and subsequent indices
    correspond to the cumulative sum of counts up to the previous entry.

    Parameters
    ----------
    counts : np.ndarray
        A 1D array of counts representing the size of each segment.

    Returns
    -------
    np.ndarray
        A 1D array of cutoff indices where each value indicates the starting index
        for the corresponding segment.
    """
    # Initialize an empty array for cutoff indices
    cutoffs = np.empty_like(counts)

    # Set the first cutoff to 0 (start of the first segment)
    cutoffs[0] = 0

    # Compute subsequent cutoffs using cumulative sums, excluding the last element
    cutoffs[1:] = counts.cumsum()[:-1]

    return cutoffs
