"""Optional Numba kernels for similarity computation."""

import numpy as np
from numba import njit

from copairs.compute import parallel_map

# Mirrors FLOAT/DOUBLE_pairwise_sum in
# numpy/_core/src/umath/loops_utils.h.src (identical in NumPy 1.26.4-2.3.5).
_PAIRWISE_BLOCK_SIZE = 128
_PAIRWISE_STACK_SIZE = 64


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
