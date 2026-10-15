"""Exact mAP null of the iid replicate design, simulated from pure-noise groups.

A group is n replicates ranked against K private controls; every profile has D iid
Gaussian features, and similarities are cosine. With no signal, the group's mAP has one
distribution for the whole design, so simulating many pure-noise groups gives its exact
null. A test that compares an observed mAP with it is calibrated by construction.
"""

import numba
import numpy as np

D = 50


@numba.njit(parallel=True, cache=True)
def group_maps(S, n):
    """Compute the mAP of each group from its members' similarities to its profiles.

    Parameters
    ----------
    S : np.ndarray
        ``(groups, n, n + K)`` similarities of each group's n members to its profiles,
        members first.
    n : int
        Members per group.
    """
    b, _, N = S.shape
    out = np.empty(b)
    for g in numba.prange(b):
        vals = np.empty(N - 1)
        lab = np.empty(N - 1, dtype=np.bool_)
        tot = 0.0
        for i in range(n):
            c = 0
            for j in range(N):
                if j != i:
                    vals[c] = -S[g, i, j]
                    lab[c] = j < n
                    c += 1
            order = np.argsort(vals)
            hits, ap = 0, 0.0
            for r in range(N - 1):
                if lab[order[r]]:
                    hits += 1
                    ap += hits / (r + 1)
            tot += ap / (n - 1)
        out[g] = tot / n
    return out


def features(n, K, b, s, rng):
    """Draw unit-norm features of b groups; members share a direction of strength s."""
    X = rng.standard_normal((b, n + K, D), dtype=np.float32) / np.float32(np.sqrt(D))
    if s:
        v = rng.standard_normal((b, 1, D), dtype=np.float32)
        X[:, :n] += np.float32(s) * v / np.linalg.norm(v, axis=2, keepdims=True)
    return X / np.linalg.norm(X, axis=2, keepdims=True)


def maps_of(n, K, total, s, rng, keep=None):
    """Simulate ``total`` groups and return their mAPs; append features to ``keep``."""
    chunk = max(1, int(2e7 // ((n + K) * D)))
    out, kept = [], []
    for b0 in range(0, total, chunk):
        X = features(n, K, min(chunk, total - b0), s, rng)
        out.append(group_maps(np.matmul(X[:, :n], X.transpose(0, 2, 1)), n))
        if keep is not None:
            kept.append(X)
    if keep is not None:
        keep.append(np.concatenate(kept))
    return np.concatenate(out)


def exact_pvalues(null_sorted, x):
    """Return exact-null p-values: the share of pure-noise groups with mAP >= x."""
    count = len(null_sorted) - np.searchsorted(null_sorted, np.asarray(x) - 1e-9)
    return (count + 1) / (len(null_sorted) + 1)
