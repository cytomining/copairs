"""Optional Numba kernels for similarity computation."""

import numpy as np
from numba import njit

from copairs.compute import parallel_map

# Mirrors FLOAT/DOUBLE_pairwise_sum in
# numpy/_core/src/umath/loops_utils.h.src (identical in NumPy 1.26.4-2.3.5).
_PAIRWISE_BLOCK_SIZE = 128
_PAIRWISE_STACK_SIZE = 64


@njit(nogil=True, fastmath=False, cache=True)
def _build_rank_lists(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
):
    """Group directed scores by query and merge class-specific rankings."""
    n_pos = pos_pairs.size
    n_neg = neg_pairs.size

    # Group each class independently by query. Keeping positive and negative
    # workspaces separate avoids the reference path's global concatenation and
    # lexsort while retaining every directed duplicate and overlapping edge.
    pos_queries = np.empty(n_pos, dtype=pos_pairs.dtype)
    pos_keys = np.empty(n_pos, dtype=pos_sims.dtype)
    for directed_i in range(n_pos):
        pair_i = directed_i // 2
        pos_queries[directed_i] = pos_pairs[pair_i, directed_i % 2]
        # Assignment to the score-typed workspace preserves NumPy's current
        # float32 rounding for the ``1 - similarity`` ordering key.
        pos_keys[directed_i] = 1.0 - pos_sims[pair_i]

    neg_queries = np.empty(n_neg, dtype=neg_pairs.dtype)
    neg_keys = np.empty(n_neg, dtype=neg_sims.dtype)
    for directed_i in range(n_neg):
        pair_i = directed_i // 2
        neg_queries[directed_i] = neg_pairs[pair_i, directed_i % 2]
        neg_keys[directed_i] = 1.0 - neg_sims[pair_i]

    pos_order = np.argsort(pos_queries)
    neg_order = np.argsort(neg_queries)
    grouped_pos_queries = pos_queries[pos_order]
    grouped_neg_queries = neg_queries[neg_order]
    grouped_pos_keys = pos_keys[pos_order]
    grouped_neg_keys = neg_keys[neg_order]

    max_queries = n_pos + n_neg
    paired_ix = np.empty(max_queries, dtype=pos_pairs.dtype)
    counts = np.empty(max_queries, dtype=np.uint32)
    rel_k_list = np.empty(max_queries, dtype=np.uint32)

    pos_start = 0
    neg_start = 0
    query_i = 0
    rel_i = 0
    while pos_start < n_pos or neg_start < n_neg:
        if neg_start >= n_neg or (
            pos_start < n_pos
            and grouped_pos_queries[pos_start] <= grouped_neg_queries[neg_start]
        ):
            query = grouped_pos_queries[pos_start]
        else:
            query = grouped_neg_queries[neg_start]

        pos_stop = pos_start
        while pos_stop < n_pos and grouped_pos_queries[pos_stop] == query:
            pos_stop += 1
        neg_stop = neg_start
        while neg_stop < n_neg and grouped_neg_queries[neg_stop] == query:
            neg_stop += 1

        # NumPy lexsort places NaNs last and is stable when both ordering keys
        # compare equal. The class-specific sorts have the same nonfinite order;
        # the merge below supplies source order as the tie-breaker, so positives
        # precede negatives for equal finite keys and when both keys are NaN.
        grouped_pos_keys[pos_start:pos_stop].sort()
        grouped_neg_keys[neg_start:neg_stop].sort()

        paired_ix[query_i] = query
        counts[query_i] = (pos_stop - pos_start) + (neg_stop - neg_start)
        query_i += 1

        pos_i = pos_start
        neg_i = neg_start
        while pos_i < pos_stop and neg_i < neg_stop:
            pos_key = grouped_pos_keys[pos_i]
            neg_key = grouped_neg_keys[neg_i]
            if np.isnan(pos_key):
                take_pos = np.isnan(neg_key)
            elif np.isnan(neg_key):
                take_pos = True
            else:
                take_pos = pos_key <= neg_key

            if take_pos:
                rel_k_list[rel_i] = 1
                pos_i += 1
            else:
                rel_k_list[rel_i] = 0
                neg_i += 1
            rel_i += 1

        while pos_i < pos_stop:
            rel_k_list[rel_i] = 1
            pos_i += 1
            rel_i += 1
        while neg_i < neg_stop:
            rel_k_list[rel_i] = 0
            neg_i += 1
            rel_i += 1

        pos_start = pos_stop
        neg_start = neg_stop

    return (
        paired_ix[:query_i].copy(),
        rel_k_list,
        counts[:query_i].copy(),
    )


def _validate_rank_list_inputs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
) -> None:
    """Validate the narrow array contract before entering compiled code."""
    pair_arrays = (("pos_pairs", pos_pairs), ("neg_pairs", neg_pairs))
    for name, pairs in pair_arrays:
        if not isinstance(pairs, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                f"{name} must be a 2-D array with exactly two columns; "
                f"got shape {pairs.shape}."
            )
        if pairs.dtype.kind not in "iu" or not pairs.dtype.isnative:
            raise TypeError(
                f"{name} must use a native signed or unsigned integer dtype; "
                f"got {pairs.dtype}. Convert it with np.asarray(..., dtype=np.int64)."
            )

    score_arrays = (("pos_sims", pos_sims), ("neg_sims", neg_sims))
    expected_lengths = (pos_pairs.shape[0], neg_pairs.shape[0])
    for (name, scores), expected_length in zip(score_arrays, expected_lengths):
        if not isinstance(scores, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if scores.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array; got shape {scores.shape}.")
        if scores.shape[0] != expected_length:
            pair_name = "pos_pairs" if name == "pos_sims" else "neg_pairs"
            raise ValueError(
                f"{name} length must equal the number of rows in {pair_name}; "
                f"got {scores.shape[0]} and {expected_length}."
            )
        if scores.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(
                f"{name} must use a native float32 or float64 dtype; "
                f"got {scores.dtype}. Convert it with "
                "np.asarray(..., dtype=np.float32)."
            )


def build_rank_lists(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
):
    """Build regular AP rank lists for validated native floating-point scores."""
    _validate_rank_list_inputs(pos_pairs, neg_pairs, pos_sims, neg_sims)
    pair_dtype = np.result_type(pos_pairs.dtype, neg_pairs.dtype)
    score_dtype = np.result_type(pos_sims.dtype, neg_sims.dtype)
    return _build_rank_lists(
        np.ascontiguousarray(pos_pairs, dtype=pair_dtype),
        np.ascontiguousarray(neg_pairs, dtype=pair_dtype),
        np.ascontiguousarray(pos_sims, dtype=score_dtype),
        np.ascontiguousarray(neg_sims, dtype=score_dtype),
    )


@njit(nogil=True, fastmath=False, cache=True)
def _cosine_pairs_range(
    normalized_feats: np.ndarray,
    pair_ix: np.ndarray,
    result: np.ndarray,
    start: int,
    stop: int,
) -> None:
    """Evaluate an indexed range with NumPy-exact pairwise reduction grouping."""
    n_features = normalized_feats.shape[1]
    # These fixed-size stacks turn NumPy's recursive pairwise reduction into an
    # iterative one. They are allocated once per batch, not once per pair. A
    # depth of 64 covers feature counts far beyond addressable array sizes.
    node_starts = np.empty(_PAIRWISE_STACK_SIZE, dtype=np.int64)
    node_lengths = np.empty(_PAIRWISE_STACK_SIZE, dtype=np.int64)
    node_states = np.empty(_PAIRWISE_STACK_SIZE, dtype=np.uint8)
    partials = np.empty(_PAIRWISE_STACK_SIZE, dtype=normalized_feats.dtype)

    for pair_i in range(start, stop):
        left = pair_ix[pair_i, 0]
        right = pair_ix[pair_i, 1]
        depth = 0
        node_starts[0] = 0
        node_lengths[0] = n_features
        node_states[0] = 0

        while depth >= 0:
            feature_start = node_starts[depth]
            length = node_lengths[depth]
            if length > _PAIRWISE_BLOCK_SIZE:
                # NumPy bisects recursively and rounds the left side down to a
                # multiple of the eight-accumulator unroll factor.
                left_length = length // 2
                left_length -= left_length % 8
                node_states[depth] = 1  # waiting for the left child
                depth += 1
                node_starts[depth] = feature_start
                node_lengths[depth] = left_length
                node_states[depth] = 0
                continue

            if length < 8:
                # The Python wrapper handles zero-width features. Starting with
                # the first product is equivalent to NumPy's typed ``-0 +
                # first`` for all IEEE values while retaining float32 dtype.
                value = (
                    normalized_feats[left, feature_start]
                    * normalized_feats[right, feature_start]
                )
                for i in range(1, length):
                    value += (
                        normalized_feats[left, feature_start + i]
                        * normalized_feats[right, feature_start + i]
                    )
            else:
                # NumPy's FLOAT/DOUBLE_pairwise_sum uses eight accumulators for
                # blocks through 128, then combines this exact tree and peels
                # the remainder. Products stay inside each add to mirror the
                # multiply-then-sum temporary without allocating it.
                r0 = (
                    normalized_feats[left, feature_start]
                    * normalized_feats[right, feature_start]
                )
                r1 = (
                    normalized_feats[left, feature_start + 1]
                    * normalized_feats[right, feature_start + 1]
                )
                r2 = (
                    normalized_feats[left, feature_start + 2]
                    * normalized_feats[right, feature_start + 2]
                )
                r3 = (
                    normalized_feats[left, feature_start + 3]
                    * normalized_feats[right, feature_start + 3]
                )
                r4 = (
                    normalized_feats[left, feature_start + 4]
                    * normalized_feats[right, feature_start + 4]
                )
                r5 = (
                    normalized_feats[left, feature_start + 5]
                    * normalized_feats[right, feature_start + 5]
                )
                r6 = (
                    normalized_feats[left, feature_start + 6]
                    * normalized_feats[right, feature_start + 6]
                )
                r7 = (
                    normalized_feats[left, feature_start + 7]
                    * normalized_feats[right, feature_start + 7]
                )

                block_stop = length - (length % 8)
                for i in range(8, block_stop, 8):
                    r0 += (
                        normalized_feats[left, feature_start + i]
                        * normalized_feats[right, feature_start + i]
                    )
                    r1 += (
                        normalized_feats[left, feature_start + i + 1]
                        * normalized_feats[right, feature_start + i + 1]
                    )
                    r2 += (
                        normalized_feats[left, feature_start + i + 2]
                        * normalized_feats[right, feature_start + i + 2]
                    )
                    r3 += (
                        normalized_feats[left, feature_start + i + 3]
                        * normalized_feats[right, feature_start + i + 3]
                    )
                    r4 += (
                        normalized_feats[left, feature_start + i + 4]
                        * normalized_feats[right, feature_start + i + 4]
                    )
                    r5 += (
                        normalized_feats[left, feature_start + i + 5]
                        * normalized_feats[right, feature_start + i + 5]
                    )
                    r6 += (
                        normalized_feats[left, feature_start + i + 6]
                        * normalized_feats[right, feature_start + i + 6]
                    )
                    r7 += (
                        normalized_feats[left, feature_start + i + 7]
                        * normalized_feats[right, feature_start + i + 7]
                    )

                value = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
                for i in range(block_stop, length):
                    value += (
                        normalized_feats[left, feature_start + i]
                        * normalized_feats[right, feature_start + i]
                    )

            # Propagate the completed leaf or subtree through the explicit
            # recursion stack, preserving NumPy's left-subtree + right-subtree
            # operation at every level.
            while True:
                if depth == 0:
                    result[pair_i] = value
                    depth = -1
                    break

                depth -= 1
                if node_states[depth] == 1:
                    partials[depth] = value
                    node_states[depth] = 2  # waiting for the right child
                    parent_start = node_starts[depth]
                    parent_length = node_lengths[depth]
                    left_length = parent_length // 2
                    left_length -= left_length % 8
                    depth += 1
                    node_starts[depth] = parent_start + left_length
                    node_lengths[depth] = parent_length - left_length
                    node_states[depth] = 0
                    break

                value = partials[depth] + value


def cosine_pairs(
    normalized_feats: np.ndarray,
    pair_ix: np.ndarray,
    batch_size: int,
    progress_bar: bool = True,
) -> np.ndarray:
    """Compute indexed cosine similarities with serial, nogil Numba kernels."""
    num_pairs = len(pair_ix)
    result = np.empty(num_pairs, dtype=np.float32)
    if num_pairs == 0:
        return result

    # Compile (or load the on-disk cache entry) before worker threads enter the
    # dispatcher concurrently. The empty range does not duplicate any work.
    n_features = normalized_feats.shape[1]
    if n_features > 0:
        _cosine_pairs_range(normalized_feats, pair_ix, result, 0, 0)

    def par_func(start: int) -> None:
        stop = min(start + batch_size, num_pairs)
        if n_features == 0:
            result[start:stop] = 0.0
        else:
            _cosine_pairs_range(normalized_feats, pair_ix, result, start, stop)

    parallel_map(
        par_func,
        np.arange(0, num_pairs, batch_size),
        progress_bar=progress_bar,
    )
    return result
