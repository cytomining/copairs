"""Fast pair similarities and counting-based average precision.

Backends are ``"cuda"`` (CuPy) and ``"numba"``; ``"auto"`` picks CUDA when a
GPU is usable. AP values match copairs' rank-list implementation to float64
rounding given the same similarities. :func:`draw_average_precisions` scores
many query-vs-reference draws at once.
"""

import numpy as np

from copairs import compute
from copairs.nulls import resolve_backend
from copairs.methods import check_method
from copairs.fastap.draws import draw_average_precisions
from copairs.fastap.ranking import ap_from_pairs, resolve_fast_backend
from copairs.fastap.similarity import FAST_METRICS, PairSimilarity

__all__ = [
    "draw_average_precisions",
    "FAST_METRICS",
    "PairSimilarity",
    "ap_from_pairs",
    "pair_similarity",
    "resolve_backend",
    "setup",
]


def setup(
    method: str,
    backend: str,
    feats,
    distance,
    batch_size: int,
    progress_bar: bool,
    on_device: bool = False,
):
    """Resolve the AP stage's backend and bind its similarity function.

    The fast AP stage runs on ``"cuda"`` or ``"numba"``; for the NumPy
    implementation use ``method="legacy"``. The returned function maps pairs
    to the similarities of their ``feats`` rows, through a kernel when
    ``distance`` has one; with ``on_device``, a CUDA kernel's similarities stay
    on the GPU for :func:`ap_from_pairs`.

    Returns
    -------
    tuple
        ``(backend, similarity)``.
    """
    check_method(method, backend)
    generic = compute.get_similarity_fn(distance, progress_bar=progress_bar)

    def similarity(pairs):
        return generic(feats, pairs, batch_size)

    if method == "legacy":
        return backend, similarity
    backend = resolve_fast_backend(backend)
    kernel = pair_similarity(np.asarray(feats), distance, backend)
    if kernel is not None:
        keep = {"as_numpy": False} if on_device and backend == "cuda" else {}

        def similarity(pairs):
            return kernel(pairs, **keep)

    return backend, similarity


def pair_similarity(feats: np.ndarray, distance, backend: str):
    """Return a ``pairs -> float32 similarities`` function, or None without a kernel."""
    if not isinstance(distance, str) or distance not in FAST_METRICS:
        return None
    sim = PairSimilarity(feats, distance)
    if backend != "cuda":
        return sim
    from copairs.fastap import cuda

    return cuda.PairSimilarity(sim.x, distance)
