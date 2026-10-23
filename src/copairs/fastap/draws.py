"""Average precision of many query-vs-reference draws, batched.

A draw has ``k`` query profiles and ``m`` reference profiles. Each query
ranks the other ``k - 1`` queries (positives) and the ``m`` references
(negatives) by cosine similarity, through the float32 key ``1 - similarity``
of copairs' rank lists; a positive ties ahead of a negative with the same
key. Similarities of a chunk of draws come from one batched
matrix product, and each query row's AP from counting the negatives above
each of its sorted positives, so no row is fully sorted.
"""

import numba
import numpy as np

from copairs.fastap import ranking
from copairs._backend import backend_module
from copairs._numba_cache import cached_njit
from copairs.fastap.ranking import _rank_key, upper_bound, ap_from_counts
from copairs.fastap.similarity import _ROW_BLOCK, _unit_rows

DEFAULT_BUDGET = 2**29


def unit_rows(feats, normalized: bool = False) -> np.ndarray:
    """float32 rows of unit norm; raises on non-finite or zero-norm rows.

    With ``normalized``, rows already have unit norm and are only checked.
    Norms are computed in float64 one block of rows at a time, so rows whose
    squared norm overflows float32 keep their direction.
    """
    feats = np.asarray(feats)
    if not normalized:
        return _unit_rows(feats, center=False, dtype=np.float32, strict=True)
    x = np.asarray(feats, dtype=np.float32)
    for start in range(0, len(x), _ROW_BLOCK):
        if not np.isfinite(x[start : start + _ROW_BLOCK]).all():
            raise ValueError("non-finite features; clean them first")
    return x


# _row_ap compiles ranking's device functions, which Numba's cache would not
# see (see copairs._numba_cache).
@cached_njit((__file__, *ranking.KERNEL_SOURCES), parallel=True)
def _row_ap(sims, k, out, n_tasks):
    """AP of each ``(draw, query)`` row of ``sims`` with shape ``(n_rows, k + m)``.

    Ranks by the AP kernels' rule: ascending float32 ``1 - similarity``, a
    positive ahead of a negative with the same key. Each of ``n_tasks``
    parallel tasks scores a block of rows with one set of scratch arrays.
    """
    n_rows, width = sims.shape
    step = (n_rows + n_tasks - 1) // n_tasks
    for task in numba.prange(n_tasks):
        pos = np.empty(k - 1, dtype=np.float32)
        hist = np.empty(k, dtype=np.int64)
        for row in range(task * step, min(n_rows, (task + 1) * step)):
            q = row % k
            s = sims[row]
            j = 0
            for i in range(k):
                if i != q:
                    pos[j] = _rank_key(s[i])
                    j += 1
            pos.sort()
            hist[:] = 0
            for e in range(k, width):
                hist[upper_bound(pos, 0, k - 1, _rank_key(s[e]))] += 1
            out[row] = ap_from_counts(hist, 0, k - 1)


def _drawn_rows(feats: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows of ``feats`` that ``idx`` uses, and ``idx`` into them.

    Returns ``feats`` and ``idx`` unchanged when every row is used. Rows are
    normalized one by one, so a subset gives the same unit rows.
    """
    used = np.zeros(len(feats), dtype=bool)
    used[idx.ravel()] = True
    if used.all():
        return feats, idx
    position = np.cumsum(used, dtype=np.int64) - 1
    return feats[used], position[idx]


def _chunk(n_draws: int, k: int, m: int, d: int, budget_bytes: int) -> int:
    per_draw = 4 * (k + m) * (d + k)
    return int(max(1, min(n_draws, budget_bytes // per_draw)))


def draw_average_precisions(
    feats,
    queries,
    references,
    normalized: bool = False,
    budget_bytes: int = DEFAULT_BUDGET,
    backend: str = "numba",
) -> np.ndarray:
    """Average precision of every query of every draw.

    Parameters
    ----------
    feats : np.ndarray
        ``(n, d)`` profiles.
    queries : np.ndarray
        ``(n_draws, k)`` row indices of each draw's queries, ``k >= 2``.
    references : np.ndarray
        ``(n_draws, m)`` row indices of each draw's references.
    normalized : bool
        Whether ``feats`` rows already have unit norm (skips normalization;
        features are still checked to be finite).

        Only the rows that some draw uses are converted, normalized and
        checked: zero-norm or non-finite rows raise ValueError when drawn.
    budget_bytes : int
        Upper bound on gathered features and similarities held at once (in
        GPU memory on the CUDA backend).
    backend : {"numba", "cuda"}
        ``"cuda"`` (``pip install 'copairs[gpu]'``) computes similarities and
        APs on the GPU, except for draws of more than
        :data:`copairs.fastap.cuda.MAX_DRAW_QUERIES` queries, which run on
        Numba. Its batched similarities sum in another order, so they match
        Numba's to float32 rounding and near-ties can rank differently.

    Returns
    -------
    np.ndarray
        ``(n_draws, k)`` float64 APs.
    """
    if backend not in ("numba", "cuda"):
        raise ValueError("backend must be either 'numba' or 'cuda'.")
    queries = np.asarray(queries, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    if queries.ndim != 2 or references.ndim != 2 or len(queries) != len(references):
        raise ValueError("queries and references must be (n_draws, k) and (n_draws, m)")
    n_draws, k = queries.shape
    m = references.shape[1]
    if k < 2:
        raise ValueError(f"need at least 2 queries per draw, got {k}")
    for idx in (queries, references):
        if idx.size and (idx.min() < 0 or idx.max() >= len(feats)):
            raise ValueError(f"indices must address the {len(feats)} rows of feats")
    idx = np.concatenate([queries, references], axis=1)
    feats, idx = _drawn_rows(np.asarray(feats), idx)
    x = unit_rows(feats, normalized)
    if backend == "cuda":
        cuda = backend_module("cuda")
        if k <= cuda.MAX_DRAW_QUERIES:
            return cuda.draw_aps(x, idx, k, budget_bytes)
    out = np.empty((n_draws, k), dtype=np.float64)
    step = _chunk(n_draws, k, m, x.shape[1], budget_bytes)
    for start in range(0, n_draws, step):
        sl = slice(start, start + step)
        rows = x[idx[sl]]  # (b, k + m, d)
        sims = np.matmul(rows[:, :k], rows.transpose(0, 2, 1))  # (b, k, k + m)
        sims = sims.reshape(-1, k + m)
        n_tasks = max(1, min(len(sims), 4 * numba.get_num_threads()))
        _row_ap(sims, k, out[sl].reshape(-1), n_tasks)
        del rows, sims  # free this chunk before gathering the next one
    return out
